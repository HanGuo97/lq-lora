import ray
import click
import torch
from tqdm.auto import trange
from collections import defaultdict
from transformers.trainer import Trainer
from transformers import RobertaForMaskedLM
from typing import Tuple, List, Dict, Any, Optional, Callable

from models import misc_utils
from models.lq_utils import (
    compute_qconfig_assignments,
    prepare_data_for_qconfig_assignments)
from models.quantization_utils import QuantConfig
from models import allocation_utils as allocation_utils_LLaMA

RoBERTa_LINEAR_PARAMETER_NAME_PATTERNS = [
    "query.weight",
    "key.weight",
    "value.weight",
    "output.dense.weight",
    "intermediate.dense.weight",
]


def _name_matches_patterns_RoBERTa(name: str) -> bool:
    return any([
        pattern in name for pattern in
        RoBERTa_LINEAR_PARAMETER_NAME_PATTERNS])


# Alias
prepare_model_inputs_for_qconfig_assignments_RoBERTa = (
    allocation_utils_LLaMA.prepare_model_inputs_for_qconfig_assignments_LLaMA)


def compute_empirical_Fisher_RoBERTa(
    model: RobertaForMaskedLM,
    inputs: List[Dict],
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if not isinstance(model, RobertaForMaskedLM):
        raise TypeError

    fisher_dict = {}
    for name, param in model.roberta.encoder.layer.named_parameters():
        if device is None:
            _device = param.device
        else:
            _device = device
        if _name_matches_patterns_RoBERTa(name):
            fisher_dict[name] = torch.zeros_like(
                param,
                dtype=dtype,
                device=_device)

    model.train()
    for i in trange(len(inputs)):
        # We need instance-level gradients
        if (inputs[i]["labels"].shape[0] != 1 or
            inputs[i]["input_ids"].shape[0] != 1):
            raise ValueError

        # Compute the gradients
        model.zero_grad()
        outputs = model(**inputs[i])
        if not isinstance(outputs, dict):
            raise NotImplementedError
        loss = outputs["loss"]
        loss.backward()

        # Accumulate the squared gradients
        for name, param in model.roberta.encoder.layer.named_parameters():
            if _name_matches_patterns_RoBERTa(name):
                if param.grad is None:
                    raise ValueError
                fisher_dict[name].add_(
                    param
                    .grad
                    .detach()
                    .pow(2)
                    .to(dtype=dtype, device=device))

    return fisher_dict


def prepare_data_for_qconfig_assignments_RoBERTa(
    model_path: str,
    inputs_path: Optional[str],
    num_ranks: Optional[int],
    num_partitions: int,
    save_path: Optional[str] = None,
) -> List[Dict]:
    # if ray.cluster_resources()["CPU"] != 80.:
    #     raise ValueError

    @ray.remote(num_cpus=10, num_gpus=1)
    def _func(_qconfigs: List[QuantConfig]) -> Dict:
        names = []
        params = []
        model = RobertaForMaskedLM.from_pretrained(model_path)
        for name, param in model.roberta.encoder.layer.named_parameters():
            if _name_matches_patterns_RoBERTa(name):
                names.append(name)
                params.append(param)

        if inputs_path is not None:
            raw_inputs: Dict[str, torch.Tensor] = torch.load(
                inputs_path,
                map_location=torch.device("cpu"))
            inputs = [
                raw_inputs[name].to(dtype=param.dtype)
                for name, param in zip(names, params)]
        else:
            inputs = None

        costs, weights = prepare_data_for_qconfig_assignments(
            parameters=params,
            qconfigs=_qconfigs,
            num_ranks=num_ranks,
            inputs=inputs)

        return {
            # meta data
            "model_path": model_path,
            "inputs_path": inputs_path,
            "num_ranks": num_ranks,
            "num_partitions": num_partitions,
            # data
            "names": names,
            "shapes": [param.shape for param in params],
            "nparams": sum([param.numel() for param in params]),
            "costs": costs,
            "weights": weights,
            "qconfigs": _qconfigs}

    qconfigs = []
    for b in [2, 3, 4, 8]:
        for b0 in [2, 3, 4, 8]:
            for b1 in ["bf16", "fp16", "fp32"]:
                for B0 in [16, 32, 64]:
                    for B1 in [16, 64, 256]:
                        qconfig = QuantConfig(
                            num_bits=b,
                            num_bits_0=b0,
                            num_bits_1=b1,
                            block_size_0=B0,
                            block_size_1=B1)
                        qconfigs.append(qconfig)

    if len(qconfigs) % num_partitions != 0:
        raise ValueError

    outputs_refs = []
    partition_size = int(len(qconfigs) / num_partitions)
    misc_utils.swarn(
        f"Preparing Data\n"
        f"- {len(qconfigs)} qconfigs\n"
        f"- {num_partitions} partitions of size {partition_size}",
        fg="blue")
    for i in range(0, len(qconfigs), partition_size):
        outputs_ref = _func.remote(qconfigs[i: i + partition_size])
        outputs_refs.append(outputs_ref)

    outputs = ray.get(outputs_refs)
    if save_path is not None:
        torch.save(outputs, save_path)
        click.secho(f"Saved to {save_path}", fg="blue")
    return outputs


FILE_NAMES_DICT = {
    "roberta-large/lpq-64/c4"  : "/export/share/experiments/20230923/ilp_data_fp32/roberta-large.ilp.ranks-64.data-True.pth",
    "roberta-large/lpq-64/None": "/export/share/experiments/20230923/ilp_data_fp32/roberta-large.ilp.ranks-64.data-False.pth",
    "roberta-large/lora/c4"    : "/export/share/experiments/20230923/ilp_data_fp32/roberta-large.ilp.ranks-None.data-True.pth",
    "roberta-large/lora/None"  : "/export/share/experiments/20230923/ilp_data_fp32/roberta-large.ilp.ranks-None.data-False.pth",
}


def create_qconfig_and_sensitivity_dict_RoBERTa(
    identifier: str,
) -> Tuple[Dict[str, QuantConfig], Dict[str, torch.Tensor]]:

    # Some special cases
    if identifier == "nf3":
        nf3_qconfig_dict = defaultdict(
            lambda: QuantConfig(
                num_bits=3,
                num_bits_0=8,
                num_bits_1="fp32",
                block_size_0=64,
                block_size_1=256))
        return nf3_qconfig_dict, {}

    if identifier == "nf4":
        nf4_qconfig_dict = defaultdict(
            lambda: QuantConfig(
                num_bits=4,
                num_bits_0=8,
                num_bits_1="fp32",
                block_size_0=64,
                block_size_1=256))
        return nf4_qconfig_dict, {}

    GIGABYTES = 1024. ** 3
    KEY_PREFIX = "base_model.model.roberta.encoder.layer."
    BUDGET_SUBSTRING = ",budget="

    file_name_identifier, budget_identifier = (
        identifier.split(BUDGET_SUBSTRING))
    budget = float(budget_identifier)

    outputs = torch.load(FILE_NAMES_DICT[file_name_identifier])
    names = outputs[0]["names"]
    num_params = outputs[0]["nparams"]
    inputs_path = outputs[0]["inputs_path"]
    if inputs_path is not None:
        _sensitivity_dict = torch.load(
            inputs_path,
            map_location=torch.device("cpu"))
    else:
        _sensitivity_dict = None

    _costs = []
    _weights = []
    qconfigs = []
    for output in outputs:
        if not all([
            output["names"] == names,
            output["nparams"] == num_params,
            output["inputs_path"] == inputs_path]):
            raise ValueError
        _costs.append(output["costs"])
        _weights.append(output["weights"])
        qconfigs.extend(output["qconfigs"])
    costs = torch.cat(_costs, dim=1)
    weights = torch.cat(_weights, dim=1)

    # Normalize the costs and weights
    normalized_costs   = costs   / torch.linalg.norm(costs) * 1000.
    normalized_budget  = budget  / GIGABYTES * num_params
    normalized_weights = weights / GIGABYTES
    assignments_cost, assignments = compute_qconfig_assignments(
        budget=normalized_budget,
        costs=normalized_costs,
        weights=normalized_weights,
        num_chunks=1)

    if not all([
        costs.shape == (len(names), len(qconfigs)),
        weights.shape == (len(names), len(qconfigs)),
        assignments.shape == (len(names), len(qconfigs))]):
        raise ValueError

    assignments_cost = (
        assignments_cost *
        torch.linalg.norm(costs) /
        1000.)
    assignments_cost_ = (
        (assignments * costs).sum())
    assignments_weight_ = (
        (assignments * weights).sum() /
        num_params)
    click.secho(
        f"[DEBUG ILP] \n"
        f"Costs:   ({assignments_cost:.3f}, {assignments_cost_:.3f} | {assignments_cost - assignments_cost_:.3e})\n"
        f"Budgets: ({budget:.3f}, {assignments_weight_:.3f} | {budget - assignments_weight_:.3e})",
        fg="yellow")

    qconfig_dict = {}
    sensitivity_dict = {}
    for i0, (i1, qconfig_index) in enumerate(assignments.nonzero().tolist()):
        if i0 != i1:
            raise ValueError
        key0 = names[i1]
        key1 = f"{KEY_PREFIX}{key0}".removesuffix(".weight")
        qconfig_dict[key1] = qconfigs[qconfig_index]
        if _sensitivity_dict is not None:
            sensitivity_dict[key1] = _sensitivity_dict[key0]

    click.secho(
        f"Created qconfig and sensitivity dict:\n"
        f"\t-identifier={identifier}\n"
        f"\t-file_name_identifier={file_name_identifier}\n"
        f"\t-budget={budget}\n"
        f"\t-sensitivity={_sensitivity_dict is not None}\n"
        f"\t-assignment_cost={assignments_cost}",
        fg="blue")
    return qconfig_dict, sensitivity_dict
