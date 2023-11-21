import torch
import bitsandbytes as bnb
from typing import Tuple, List, Dict, Union
from models.tensor_container_utils import QuantizedTensor
from models.quantization_utils import QuantConfig
from models.quantization_utils_2 import quantize


def parse_time(profile: torch.profiler.profile) -> Dict[str, float]:
    time_dict = {}
    time_lines = profile.key_averages().table().splitlines()[-2:]
    for line in time_lines:
        raw_key, raw_value = line.split(": ")
        if raw_key == "Self CPU time total":
            key = "CPU"
        elif raw_key == "Self CUDA time total":
            key = "GPU"
        else:
            raise ValueError
        if raw_value.endswith("ms"):
            value = float(raw_value.removesuffix("ms"))
        elif raw_value.endswith("us"):
            raise ValueError
        elif raw_value.endswith("s"):
            if "ms" in raw_value or "us" in raw_value:
                raise ValueError(f"Raw Value = {raw_value}")
            value = float(raw_value.removesuffix("s")) * 1000.
        else:
            raise ValueError
        time_dict[key] = value
    return time_dict


def make_toy_task(
    dim0: int,
    dim1: int,
    num_matrices: int,
) -> Tuple[List[torch.Tensor],
           List[torch.Tensor],
           List[bnb.nn.Params4bit],
           Dict[int, List[QuantizedTensor]],
           Dict[int, List[QuantizedTensor]]]:

    Xs = [
        torch.rand(dim0, dim1, device="cuda")
        for _ in range(num_matrices)
    ]

    As = [
        torch.rand(dim1, dim1, device="cuda")
        for _ in range(num_matrices)
    ]

    qAs_bnb = [
        # `bitsandbytes` quantize matrices
        # during the CPU->GPU data movement.
        bnb.nn.Params4bit(  # type: ignore
            A.to("cpu"),
            requires_grad=False,
            compress_statistics=True,
            quant_type="nf4")
        .to("cuda")
        for A in As
    ]

    qAs_slow_dict = {}
    qAs_fast_dict = {}
    for b in [2, 3, 4, 8]:

        qAs_slow_dict[b] = [
            quantize(
                A,
                method="blockwise-nf",
                qconfig=QuantConfig(
                    num_bits=b,
                    num_bits_0=8,
                    num_bits_1="fp32",
                    block_size_0=64,
                    block_size_1=256),
                legacy=True)
            for A in As
        ]

        qAs_fast_dict[b] = [
            quantize(
                A,
                method="blockwise-nf",
                qconfig=QuantConfig(
                    num_bits=b,
                    num_bits_0=8,
                    num_bits_1="fp32",
                    block_size_0=64,
                    block_size_1=256),
                legacy=False)
            for A in As
        ]

    return Xs, As, qAs_bnb, qAs_slow_dict, qAs_fast_dict


def measure_times(
    Xs: List[torch.Tensor],
    As: Union[List[torch.Tensor],
              List[bnb.nn.Params4bit],
              List[QuantizedTensor]],
    is_bnb: bool = False,
) -> Dict[str, float]:
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:

        if is_bnb is False:
            with torch.no_grad():
                for X, A in zip(Xs, As):
                    O = torch.mm(X, A)
        else:
            with torch.no_grad():
                for X, A in zip(Xs, As):
                    O = bnb.matmul_4bit(
                        X, A,
                        bias=None,
                        quant_state=A.quant_state)

    return parse_time(p)


def run_experiments() -> List[Dict]:
    time_dicts = []
    for dim1 in [4096]:
        for dim0 in [1, 16, dim1]:
            Xs, As, qAs_bnb, qAs_slow_dict, qAs_fast_dict = make_toy_task(
                dim0=dim0,
                dim1=dim1,
                num_matrices=100)

            time_dict_dense = measure_times(Xs, As)
            time_dict_dense["dim1"] = dim1
            time_dict_dense["dim0"] = dim0
            time_dict_dense["method"] = "dense"
            time_dicts.append(time_dict_dense)

            time_dict_bnb = measure_times(Xs, qAs_bnb, is_bnb=True)
            time_dict_bnb["dim0"] = dim0
            time_dict_bnb["dim1"] = dim1
            time_dict_bnb["method"] = "bnb"
            time_dicts.append(time_dict_bnb)

            for b in [2, 3, 4, 8]:
                time_dict_slow = measure_times(Xs, qAs_slow_dict[b])
                time_dict_slow["dim0"] = dim0
                time_dict_slow["dim1"] = dim1
                time_dict_slow["method"] = f"{b}bit-legacy"
                time_dicts.append(time_dict_slow)

                time_dict_fast = measure_times(Xs, qAs_fast_dict[b])
                time_dict_fast["dim0"] = dim0
                time_dict_fast["dim1"] = dim1
                time_dict_fast["method"] = f"{b}bit"
                time_dicts.append(time_dict_fast)

            del Xs, As, qAs_bnb, qAs_slow_dict, qAs_fast_dict

    return time_dicts
