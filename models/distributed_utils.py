import torch
import click
import optree
from copy import deepcopy
from torch import distributed as dist
from peft import PeftModelForCausalLM
from transformers import TrainingArguments
from transformers.training_args import ParallelMode
from typing import Union, Any

from models import tensor_container_utils


def maybe_prepare_model_for_ddp(
    args: TrainingArguments,
    model: PeftModelForCausalLM,
) -> None:
    # no need to do anything if we are not using DDP
    if args.parallel_mode != ParallelMode.DISTRIBUTED:
        return

    # no need to do anything if we are not using PeftModelForCausalLM
    if not isinstance(model, PeftModelForCausalLM):
        return

    click.secho(f"-------------- Running DDP with {type(model)} --------------", fg="yellow")
    # Manually sync the `QuantizedTensor` across all processes, note that current
    # implementation is somewhat inefficient because we do another broadcast after this.
    for name, param in model.named_parameters():
        if type(param) is tensor_container_utils.QuantizedTensor:
            state_dict = param._container.to_dict()
            state_dict_broadcast = optree.tree_map(
                _broadcast,
                state_dict,
                none_is_leaf=True)
            param_broadcast = type(param)(
                container=param._container.from_dict(state_dict_broadcast),
                tensor_like=param,
                transpose=param._transpose,
                compute_dtype=param._compute_dtype)
            # This is an inplace operation
            param.copy_(param_broadcast)

    # `QuantizedTensor` is not compatible with DDP, hence we need to hide them from DDP.
    # This fix is based on the following discussion:
    # - https://github.com/pytorch/pytorch/pull/44826
    # - https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L2138
    # - https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py#L135
    params_to_ignore = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if type(param) is tensor_container_utils.QuantizedTensor:
                formatted_param_name = f"{module_name}.{param_name}"
                params_to_ignore.append(formatted_param_name)
                click.secho(f"[DDP Ignore]: {formatted_param_name}", fg="yellow")

                # DDP syncs all parameters and buffers at the beginning.
                # Since we are hiding the quantized tensors from DDP,
                # we need to assert that these tensors are equal.
                state_dict = param._container.to_dict()
                optree.tree_map_with_path(
                    _assert_all_equal,
                    state_dict,
                    none_is_leaf=True)

    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
        module=model,
        params_and_buffers_to_ignore=params_to_ignore)


def _broadcast(maybe_tensor: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    if isinstance(maybe_tensor, torch.Tensor):
        dist.broadcast(maybe_tensor, src=0)
        return maybe_tensor
    else:
        object_list = [maybe_tensor]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]


def _assert_all_equal(name: str, maybe_tensor: Union[torch.Tensor, Any]) -> None:
    if isinstance(maybe_tensor, torch.Tensor):
        # create temporary buffer to store the broadcasted data
        if dist.get_rank() == 0:
            maybe_tensor_ = maybe_tensor.clone().detach()
        else:
            maybe_tensor_ = torch.empty_like(maybe_tensor)

        # broadcast and assert
        dist.broadcast(maybe_tensor_, src=0)
        if not (maybe_tensor == maybe_tensor_).all():
            raise ValueError(
                f"[{name}]: original != broadcasted tensor:\t"
                f"{maybe_tensor} != {maybe_tensor_}")
    else:
        # https://github.com/pytorch/pytorch/issues/56142
        if dist.get_rank() == 0:
            object_list = [deepcopy(maybe_tensor)]
        else:
            object_list = [None]

        # broadcast and assert
        dist.broadcast_object_list(object_list, src=0)
        if maybe_tensor != object_list[0]:
            raise ValueError(
                f"[{name}]: original != broadcasted tensor:\t"
                f"{maybe_tensor} != {object_list[0]}")
