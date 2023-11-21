import os
import math
import torch
import click
from transformers import (
    GPTQConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    RobertaForSequenceClassification)
from transformers.trainer import Trainer
from torch.utils import _pytree as pytree
from peft.tuners import lora
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    get_peft_model,
    prepare_model_for_kbit_training)
from typing import List, Optional, Union, Dict, Any, cast

from models import misc_utils
from models import allocation_utils as allocation_utils_LLaMA
from models import allocation_utils_2 as allocation_utils_RoBERTa
from models import tensor_container_utils
from models.lq_utils import (
    QuantConfig,
    maybe_sparsify_or_quantize,
    lowrank_quantized_sparse_decomposition_maybe_cast)


def get_hf_quantization_config(method: Optional[str], sequence_length: int) -> Dict[str, Any]:

    if method is None:
        return {}

    if method == "bnb-4bit":
        click.secho("Loading in 4-bit", fg="red")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        return {"device_map": "auto",
                "quantization_config": bnb_config}

    if method == "gptq-3bit":
        click.secho("Loading in GPTQ 3-bit", fg="red")
        gptq_config = GPTQConfig(
            bits=3,
            dataset="c4",
            model_seqlen=sequence_length,
            disable_exllama=True)
        return {"device_map": "auto",
                "quantization_config": gptq_config}

    if method == "gptq-4bit":
        click.secho("Loading in GPTQ 4-bit", fg="red")
        gptq_config = GPTQConfig(
            bits=4,
            dataset="c4",
            model_seqlen=sequence_length,
            disable_exllama=True)
        return {"device_map": "auto",
                "quantization_config": gptq_config}

    raise ValueError(f"Unknown quantization method: {method}")


def save_full_model(trainer: Trainer) -> None:
    if not isinstance(trainer.model, (PeftModelForCausalLM, PeftModelForSequenceClassification)):
        raise TypeError(
            f"Expected `PeftModelForCausalLM`, or "
            f"`PeftModelForSequenceClassification`, "
            f"but got {type(trainer.model)}")
    if not trainer.args.should_save:
        return

    state_dict = trainer.model.state_dict()
    file_name = os.path.join(
        trainer.args.output_dir,
        "full_model.pth")
    torch.save(state_dict, file_name)
    click.secho(f"Saved model state dict to {file_name}", fg="green")


def load_peft_model(
    model: LlamaForCausalLM,
    checkpoint_dir: str,
    checkpoint_preprocess_embedding: bool = False,
) -> PeftModelForCausalLM:
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(f"Expected LlamaForCausalLM, but got {type(model)}")

    peft_model = PeftModelForCausalLM.from_pretrained(
        model=model,
        model_id=checkpoint_dir,
        is_trainable=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        "full_model.pth")
    # We need to load the state dict to CUDA instead of CPU.
    # This is because for `QuantizedTensor`, `tensor.device`
    # is not reflective of the actual underlying device. Rather,
    # we set this attribute to the device of the tensor during
    # creation/quantization (which is usually GPU). Unfortunately,
    # loading the tensor to CPU will not update this attribute.
    state_dict = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda"))
    # The above command makes sure that the underlying tensors are on GPU, and
    # the folllowing command asserts that the tensor "behaves" as if it's on GPU
    if pytree.tree_all_only(
        tensor_container_utils.QuantizedTensor,
        lambda qtensor: qtensor.device.type == "cuda",
        state_dict) is False:
        raise ValueError

    if checkpoint_preprocess_embedding is True:
        # LLaMA-specific ad-hoc fix.
        state_dict = _checkpoint_handle_mismatched_embedding_shape_for_LLaMA(
            model=peft_model,
            state_dict=state_dict)

    # Unfortunately, `load_state_dict` does in-place copy behind the scene, but we
    # cannot in-place copy a `QuantizedTensor` into a `torch.Tensor`. Instead, we will
    # first do out-of-place assignment (so they are compatible), before in-place copy.
    _transform_lora_layers_for_loading(
        model=peft_model,
        state_dict=state_dict)

    # Note: PyTorch 2.1 has a new feature that allows us to load a state dict via assignment,
    # `new_model.load_state_dict(state_dict, assign=True)`. Unfortunately, this is still 
    # bit fragile, so we will do the old-school way instead.
    peft_model.load_state_dict(state_dict)

    click.secho(
        f"Loaded \n"
        f"- PEFT model from {checkpoint_dir}\n"
        f"- state dict from {checkpoint_path}",
        fg="green")

    # Mostly to get the type-checker to stop complaining
    if not isinstance(peft_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(peft_model)}")
    return peft_model


def _transform_lora_layers_for_loading(
    model: PeftModelForCausalLM,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError
    for name, submodule in model.named_modules():
        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):
            if type(submodule) is lora.Linear:
                qweight = state_dict[f"{name}.weight"]
                if not isinstance(
                    qweight,
                    tensor_container_utils.QuantizedTensor):
                    raise TypeError
                replace_weight_(
                    module=submodule,
                    new_weight=qweight)
            else:
                raise TypeError


def _checkpoint_handle_mismatched_embedding_shape_for_LLaMA(
    model: PeftModelForCausalLM,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError
    if not isinstance(model.base_model.model, LlamaForCausalLM):
        raise TypeError

    lm_head_weight_ = state_dict["base_model.model.lm_head.weight"]
    lm_head_weight = model.base_model.model.lm_head.weight
    lm_head_weight = lm_head_weight.to(device=lm_head_weight_.device)
    embed_tokens_weight_ = state_dict["base_model.model.model.embed_tokens.weight"]
    embed_tokens_weight = model.base_model.model.model.embed_tokens.weight
    embed_tokens_weight = embed_tokens_weight.to(device=embed_tokens_weight_.device)

    if lm_head_weight.shape != (
        lm_head_weight_.shape[0] + 1,
        lm_head_weight_.shape[1]):
        raise ValueError
    if embed_tokens_weight.shape != (
        embed_tokens_weight_.shape[0] + 1,
        embed_tokens_weight_.shape[1]):
        raise ValueError
    if not (lm_head_weight[:-1, :] == lm_head_weight_).all():
        raise ValueError
    if not (embed_tokens_weight[:-1, :] == embed_tokens_weight_).all():
        raise ValueError

    state_dict["base_model.model.lm_head.weight"] = lm_head_weight
    state_dict["base_model.model.model.embed_tokens.weight"] = embed_tokens_weight
    return state_dict


def _enable_gradient_checkpointing(model: LlamaForCausalLM) -> None:
    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()


def prepare_model_for_lora(
    model: LlamaForCausalLM,
    num_ranks: int,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_preprocess_embedding: bool = False,
) -> PeftModelForCausalLM:

    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(f"Expected LlamaForCausalLM, but got {type(model)}")
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    peft_config = LoraConfig(
        r=num_ranks,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM")

    # This function is useful even if we are not doing int8 training.
    # (1) Freezing all the parameters before we add LoRA.
    # (2) Casting all fp16/bf16 parameters to fp32.
    # (3) Including gradient checkpointing when the model is loaded in 4/8-bit and some details.
    new_model = prepare_model_for_kbit_training(
        model=model,
        use_gradient_checkpointing=use_gradient_checkpointing)
    # The above only enables gradient checkpointing for BNB-quantized models
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L81C1-L117C17
    if use_gradient_checkpointing is True:
        _enable_gradient_checkpointing(new_model)
    if checkpoint_dir is not None:
        click.secho(
            f"Loading PEFT model from {checkpoint_dir}. "
            f"Aforementioned arguments will be ignored",
            fg="blue")
        new_model = load_peft_model(
            model=new_model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_preprocess_embedding=checkpoint_preprocess_embedding)
    else:
        new_model = get_peft_model(new_model, peft_config)
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(new_model)}")
    return new_model


def prepare_model_for_lora_classification(
    model: RobertaForSequenceClassification,
    num_ranks: int,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
) -> PeftModelForSequenceClassification:

    if not isinstance(model, RobertaForSequenceClassification):
        raise TypeError(f"Expected RobertaForSequenceClassification, but got {type(model)}")
    if target_modules is None:
        target_modules = [
            "query",
            "key",
            "value",
            "output.dense",
            "intermediate.dense",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    peft_config = LoraConfig(
        r=num_ranks,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS")

    # This function is useful even if we are not doing int8 training.
    # (1) Freezing all the parameters before we add LoRA.
    # (2) Casting all fp16/bf16 parameters to fp32.
    # (3) Including gradient checkpointing when the model is loaded in 4/8-bit and some details.
    new_model = prepare_model_for_kbit_training(
        model=model,
        use_gradient_checkpointing=use_gradient_checkpointing)
    # The above only enables gradient checkpointing for BNB-quantized models
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L81C1-L117C17
    if use_gradient_checkpointing is True:
        _enable_gradient_checkpointing(new_model)
    new_model = get_peft_model(new_model, peft_config)
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForSequenceClassification):
        raise TypeError(f"Expected PeftModelForSequenceClassification, but got {type(new_model)}")
    return new_model


def transform_lora_layers(
    lpq: bool,
    model: Union[PeftModelForCausalLM, PeftModelForSequenceClassification],
    model_name: str,
    num_iterations: int = 100,
    num_oversampling: int = 10,
    randomized: bool = True,
    device: Optional[torch.device] = None,
) -> None:

    click.secho(
        f"Transforming LoRA layers with the following configurations:\n"
        f"\t -lpq: {lpq}\n"
        f"\t -model_name: {model_name}\n"
        f"\t -num_iterations: {num_iterations}\n"
        f"\t -num_oversampling: {num_oversampling}\n"
        f"\t -randomized: {randomized}",
        fg="blue")

    if isinstance(model, PeftModelForCausalLM):
        qconfig_dict, sensitivity_dict = (
            allocation_utils_LLaMA
            .create_qconfig_and_sensitivity_dict_LLaMA(
                identifier=model_name))
    elif isinstance(model, PeftModelForSequenceClassification):
        qconfig_dict, sensitivity_dict = (
            allocation_utils_RoBERTa
            .create_qconfig_and_sensitivity_dict_RoBERTa(
                identifier=model_name))
    else:
        raise NotImplementedError(f"Unknown model type: {type(model)}")

    for name, submodule in model.named_modules():

        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):

            # These operations will be too slow on CPU
            if device is not None:
                # This is in-place
                submodule.to(device=device)

            assert_lora_Linear_layer(submodule)
            qconfig = qconfig_dict[name]

            if lpq is False:
                print(f"{name:<50}\tqconfig={qconfig}")
                transform_lora_layer(
                    submodule,
                    qconfig=qconfig)
            else:
                num_ranks = cast(
                    int,
                    submodule.r[submodule.active_adapter])
                # Possibly empty
                sensitivity = sensitivity_dict.get(name, None)

                print(f"{name:<50}\tqconfig={qconfig}")
                transform_lora_layer_lpq(
                    submodule,
                    num_ranks=num_ranks,
                    num_iterations=num_iterations,
                    num_oversampling=num_oversampling,
                    randomized=randomized,
                    qconfig=qconfig,
                    W=sensitivity,
                    heuristic="two-sided")


@torch.no_grad()
def transform_lora_layer_lpq(
    module: lora.LoraLayer,
    num_ranks: int,
    num_iterations: int,
    num_oversampling: int,
    randomized: bool,
    qconfig: Optional[QuantConfig],
    W: Optional[torch.Tensor] = None,
    heuristic: Optional[str] = None,
) -> None:

    if type(module) is lora.Linear:
        L1, L2, Q, _ = lowrank_quantized_sparse_decomposition_maybe_cast(
            module.weight,
            num_ranks=num_ranks,
            num_iterations=num_iterations,
            num_oversampling=num_oversampling,
            randomized=randomized,
            qconfig=qconfig,
            W=W,
            heuristic=heuristic)
        replace_weight_(
            module=module,
            new_weight=Q)
    else:
        raise TypeError

    # The LoRA layer essentially does the following computation:
    # ```
    # 1. x_ = dropout(x)
    # 2. y  = x @ W.T + s * x_ @ A.T @ B.T
    # When dropout is turned off,
    #    y = x @  W.T +  s * x @ A.T @          B.T
    #      = x @ (W.T +  s *     A.T @          B.T)
    #      = x @ (W   +  s *     B   @          A  ).T
    #      = x @ (W   + [sqrt(s) B]  @ [sqrt(s) A] ).T
    # ```
    # Since LPQ applies the following computation: `W + L1 @ L2`, we want
    # ```
    # 1. L1 = sqrt(s) B
    # 2. L2 = sqrt(s) A
    # ```
    # Hence we assign
    # ```
    # 1. A = L2 / sqrt(s)
    # 2. B = L1 / sqrt(s)
    # ```
    scale_sqrt = math.sqrt(module.scaling[module.active_adapter])
    module.lora_A[module.active_adapter].weight.copy_(L2 / scale_sqrt)
    module.lora_B[module.active_adapter].weight.copy_(L1 / scale_sqrt)


@torch.no_grad()
def transform_lora_layer(
    module: lora.LoraLayer,
    qconfig: Optional[QuantConfig],
) -> None:

    if type(module) is lora.Linear:
        Q = maybe_sparsify_or_quantize(
            module.weight,
            qconfig=qconfig)
        replace_weight_(
            module=module,
            new_weight=Q)
    else:
        raise TypeError


def assert_lora_Linear_layer(
    module: lora.Linear,
) -> None:
    if type(module) is not lora.Linear:
        raise TypeError
    if module.fan_in_fan_out:
        raise ValueError
    if (len(module.lora_embedding_A) != 0 or
        len(module.lora_embedding_B) != 0):
        raise ValueError
    if (module.bias is not None or
        module.lora_A[module.active_adapter].bias is not None or
        module.lora_B[module.active_adapter].bias is not None):
        raise ValueError

    lora_B_weight = module.lora_B[module.active_adapter].weight
    lora_B_weight_all_zeros = (lora_B_weight == 0.).all().item()
    if not lora_B_weight_all_zeros:
        raise ValueError("Expected `module.lora_B.weight` to be zero.")


def replace_weight_(
    module: lora.Linear,
    new_weight: Union[torch.Tensor, tensor_container_utils.QuantizedTensor],
) -> None:
    if isinstance(new_weight, tensor_container_utils.QuantizedTensor):
        if not isinstance(module.weight, torch.nn.Parameter):
            raise TypeError
        if module.weight.requires_grad is not False:
            raise ValueError
        module.weight = torch.nn.Parameter(
            new_weight,
            requires_grad=module.weight.requires_grad)
    else:
        module.weight.copy_(new_weight)
