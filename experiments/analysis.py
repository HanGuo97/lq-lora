import torch
from tqdm.auto import tqdm
from typing import Tuple, Dict, List, Optional

from transformers import LlamaForCausalLM
from models import allocation_utils as allocation_utils_LLaMA
from experiments.prepare_ilp_and_fisher_data import MODEL_PATHS_DICT
from models.lq_utils import (
    QuantConfig,
    svd_decomposition,
    weighted_svd_decomposition,
    maybe_sparsify_or_quantize,
    lowrank_quantized_sparse_decomposition_maybe_cast)


DEFAULT_RANK = 64
DEFAULT_QCONFIGS = {
    "nf3": QuantConfig(
        num_bits=3,
        num_bits_0=8,
        num_bits_1="fp32",
        block_size_0=64,
        block_size_1=256),
    "nf4": QuantConfig(
        num_bits=4,
        num_bits_0=8,
        num_bits_1="fp32",
        block_size_0=64,
        block_size_1=256),
}


def get_error_dict(A: torch.Tensor, num_ranks: int) -> Dict[Tuple[Optional[int], int], float]:
    error_dict = {}
    for use_lpq in [True]:
        for qconfig_name in ["nf3", "nf4"]:
            qconfig = DEFAULT_QCONFIGS[qconfig_name]
            if use_lpq is False:
                A_ = maybe_sparsify_or_quantize(A, qconfig=qconfig)
            else:
                L1, L2, S, _ = lowrank_quantized_sparse_decomposition_maybe_cast(
                    A,
                    num_ranks=num_ranks,
                    qconfig=qconfig)
                A_ = torch.addmm(S, L1, L2)

            error_dict[(use_lpq, num_ranks, qconfig_name)] = (torch.linalg.norm(A - A_, ord="fro") ** 2).item()
    return error_dict


@torch.no_grad()
def analyze_weighted_svd_heuristics(
    As: List[torch.Tensor],
    Ws: List[torch.Tensor],
    qconfig: QuantConfig,
    num_ranks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    errors_lpq = torch.zeros(len(As), 5)
    errors_svd = torch.zeros(len(As), 5)
    errors_rsvd = torch.zeros(len(As), 5)
    heuristics_options = ["none", "row", "col", "two-sided"]

    for i0, (A, W) in enumerate(zip(tqdm(As), Ws)):
        A = A.cuda()
        W = W.cuda()
        for i1, heuristic in enumerate(heuristics_options):
            L1_lpq, L2_lpq, Q_and_S_lpq, _ = lowrank_quantized_sparse_decomposition_maybe_cast(
                A=A,
                num_ranks=num_ranks,
                qconfig=qconfig,
                W=W,
                heuristic=heuristic)
            L1_svd, L2_svd = weighted_svd_decomposition(
                A=A,
                W=W,
                heuristic=heuristic,
                randomized=False,
                num_ranks=num_ranks,
                num_oversampling=10)
            L1_rsvd, L2_rsvd = weighted_svd_decomposition(
                A=A,
                W=W,
                heuristic=heuristic,
                randomized=True,
                num_ranks=num_ranks,
                num_oversampling=10)
            A_lpq  = L1_lpq  @ L2_lpq + Q_and_S_lpq
            A_svd  = L1_svd  @ L2_svd
            A_rsvd = L1_rsvd @ L2_rsvd
            errors_lpq[i0 , i1] = torch.linalg.norm(torch.sqrt(W) * (A - A_lpq) , ord="fro") ** 2
            errors_svd[i0 , i1] = torch.linalg.norm(torch.sqrt(W) * (A - A_svd) , ord="fro") ** 2
            errors_rsvd[i0, i1] = torch.linalg.norm(torch.sqrt(W) * (A - A_rsvd), ord="fro") ** 2

        L1_lpq_oracle, L2_lpq_oracle, Q_and_S_lpq_oracle, _ = lowrank_quantized_sparse_decomposition_maybe_cast(
            A=torch.sqrt(W) * A,
            num_ranks=num_ranks,
            qconfig=qconfig)
        L1_svd_oracle, L2_svd_oracle = svd_decomposition(
            A=torch.sqrt(W) * A,
            randomized=False,
            num_ranks=num_ranks,
            num_oversampling=10)
        L1_rsvd_oracle, L2_rsvd_oracle = svd_decomposition(
            A=torch.sqrt(W) * A,
            randomized=True,
            num_ranks=num_ranks,
            num_oversampling=10)
        A_lpq_oracle  = L1_lpq_oracle  @ L2_lpq_oracle + Q_and_S_lpq_oracle
        A_svd_oracle  = L1_svd_oracle  @ L2_svd_oracle
        A_rsvd_oracle = L1_rsvd_oracle @ L2_rsvd_oracle
        errors_lpq[i0 , len(heuristics_options)] = torch.linalg.norm(torch.sqrt(W) * A - A_lpq_oracle , ord="fro") ** 2
        errors_svd[i0 , len(heuristics_options)] = torch.linalg.norm(torch.sqrt(W) * A - A_svd_oracle , ord="fro") ** 2
        errors_rsvd[i0, len(heuristics_options)] = torch.linalg.norm(torch.sqrt(W) * A - A_rsvd_oracle, ord="fro") ** 2

    return errors_lpq, errors_svd, errors_rsvd


@torch.no_grad()
def experiment_1() -> List[Dict]:
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATHS_DICT["llama-2-7b"],
        low_cpu_mem_usage=True)
    lora_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]

    outputs_dict = []
    for name, param in tqdm(model.named_parameters()):
        if not name.endswith(".weight"):
            raise ValueError

        if not any([tname in name for tname in lora_target_modules]):
            continue
        
        layer_index = int(name.split(".")[2])
        layer_type = name.split(".")[-2]
        if layer_type not in lora_target_modules:
            raise ValueError

        outputs_dict.append({
            "layer_index": layer_index,
            "layer_type": layer_type,
            "name": name,
            "param": param,
            # **get_error_dict(param.to("cuda"), num_ranks=32),
            **get_error_dict(param.to("cuda"), num_ranks=64),
            # **get_error_dict(param.to("cuda"), num_ranks=128),
        })
        print(name)

    return outputs_dict


@torch.no_grad()
def experiment_2() -> List[Dict]:
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATHS_DICT["llama-2-7b"],
        low_cpu_mem_usage=True)
    state_dict = model.state_dict()
    state_dict_keys = [
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.12.self_attn.v_proj.weight",
        "model.layers.20.mlp.gate_proj.weight"
    ]
    _, sensitivity_dict = (
        allocation_utils_LLaMA
        .create_qconfig_and_sensitivity_dict_LLaMA(
            identifier="llama-2-7b/lpq-64/c4,budget=2.75"))

    outputs_dict = []
    for index, name in enumerate(tqdm(state_dict_keys)):
        for use_sensitivity in [True, False]:
            if use_sensitivity is True:
                sensitivity_name = f"base_model.model.{name}".removesuffix(".weight")
                sensitivity = sensitivity_dict[sensitivity_name]
                sensitivity = sensitivity.to("cuda")
                if sensitivity.dtype != torch.float32:
                    raise TypeError
                if state_dict[name].dtype != torch.float32:
                    raise TypeError
            else:
                sensitivity = None

            # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _, _, _, errors = lowrank_quantized_sparse_decomposition_for_analysis(
                state_dict[name].to("cuda"),
                num_ranks=DEFAULT_RANK,
                randomized=True,
                qconfig=DEFAULT_QCONFIGS["nf3"],
                W=sensitivity,
                heuristic="two-sided")
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            outputs_dict.append({
                "index": index,
                "errors": errors,
                "time": time,
                "use_sensitivity": use_sensitivity,
            })

    return outputs_dict


def lowrank_quantized_sparse_decomposition_for_analysis(
    A: torch.Tensor,
    num_ranks: int,
    num_iterations: int = 100,
    num_oversampling: int = 10,
    randomized: bool = True,
    qconfig: Optional[QuantConfig] = None,
    W: Optional[torch.Tensor] = None,
    heuristic: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[float]]:

    if len(A.shape) != 2:
        raise ValueError
    if num_iterations < 1:
        raise ValueError

    errors = []
    Q = torch.zeros_like(A)
    for _ in range(num_iterations):
        if W is None:
            L1, L2 = svd_decomposition(
                A - Q,
                randomized=randomized,
                num_ranks=num_ranks,
                num_oversampling=num_oversampling)
        else:
            L1, L2 = weighted_svd_decomposition(
                A - Q,
                W=W,
                heuristic=heuristic,
                randomized=randomized,
                num_ranks=num_ranks,
                num_oversampling=num_oversampling)

        Q = maybe_sparsify_or_quantize(
            A - L1 @ L2,
            qconfig=qconfig)

        A_ = L1 @ L2 + Q
        if W is None:
            error = torch.linalg.norm(A - A_, ord="fro")
        else:
            error = torch.linalg.norm(torch.sqrt(W) * (A - A_), ord="fro")
        errors.append(error.item())

    return L1, L2, Q, errors
