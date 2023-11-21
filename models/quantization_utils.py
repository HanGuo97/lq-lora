import math
import torch
import scipy.special
import bitsandbytes as bnb
from pytorch_quantization import tensor_quant
from typing import Tuple, Optional, NamedTuple

from models import misc_utils


NF4_OFFSET = 0.9677083  # Magic number?

class QuantScheme(NamedTuple):
    values: torch.Tensor
    boundaries: torch.Tensor


class QuantConfig(NamedTuple):
    num_bits: int
    num_bits_0: int
    num_bits_1: str
    block_size_0: int
    block_size_1: int


def create_quantization_scheme(
    values: torch.Tensor,
    device: torch.device,
) -> QuantScheme:
    inf_tensor = torch.tensor([torch.inf])
    boundaries = (values[1:] + values[:-1]) / 2.
    boundaries = torch.cat([-inf_tensor, boundaries, inf_tensor], dim=0)

    values = values.to(device=device)
    boundaries = boundaries.to(device=device)
    if values.ndim != 1 or boundaries.ndim != 1:
        raise ValueError
    if values.shape[0] != boundaries.shape[0] - 1:
        raise ValueError
    return QuantScheme(
        values=values,
        boundaries=boundaries)


def quantize_with_scheme(
    A: torch.Tensor,
    qscheme: QuantScheme,
    scales_q: torch.Tensor,
    scales_dq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.shape != scales_q.shape:
        raise ValueError
    if A.shape != scales_dq.shape:
        raise ValueError
    A_scaled = A / scales_q
    # `-1` because this function assigns to the right bucket
    A_quantized = torch.bucketize(
        A_scaled,
        qscheme.boundaries,
        right=False) - 1
    A_dequantized = qscheme.values[A_quantized]
    A_dequantized = A_dequantized * scales_dq
    return A_quantized, A_dequantized


def create_normal_float_scheme(
    num_bits: int,
    device: torch.device,
) -> QuantScheme:
    # This is essentially what NF4 does.
    sigma = -1. / (
        math.sqrt(2) *
        scipy.special.erfinv(1 - 2 * NF4_OFFSET))
    qdist = torch.distributions.normal.Normal(
        loc=0.,
        scale=sigma)

    quantiles_left = torch.linspace(
        1. - NF4_OFFSET,
        0.5,
        2 ** (num_bits - 1))
    quantiles_right = torch.linspace(
        0.5,
        NF4_OFFSET,
        2 ** (num_bits - 1) + 1)
    # remove the duplicated `0.5`
    quantiles = torch.cat([
        quantiles_left[:-1],
        quantiles_right],
        dim=0)
    values = qdist.icdf(quantiles)
    return create_quantization_scheme(
        values=values,
        device=device)


def create_8bit_float_scheme(device: torch.device) -> QuantScheme:
    values = bnb.functional.create_dynamic_map()
    return create_quantization_scheme(
        values=values,
        device=device)


def quantize_with_scheme_2(
    A: torch.Tensor,
    scales: torch.Tensor,
    num_bits: int,
    dtype: str,
) -> torch.Tensor:
    if dtype not in ["nf", "fp", "int", "uint"]:
        raise ValueError

    # NVIDIA backend
    if dtype == "int":
        return tensor_quant.fake_tensor_quant(
            A,
            scales,
            num_bits)

    if dtype == "uint":
        return tensor_quant.fake_tensor_quant(
            A,
            scales,
            num_bits,
            True)  # unsigned

    # Custom backend
    if dtype == "nf":
        qscheme = create_normal_float_scheme(
            num_bits=num_bits,
            device=A.device)
    else:
        if num_bits != 8:
            raise ValueError
        qscheme = create_8bit_float_scheme(
            device=A.device)

    _, A_dequantized = quantize_with_scheme(
        A,
        qscheme=qscheme,
        scales_q=scales,
        scales_dq=scales)
    return A_dequantized


def quantize_with_nf4_original(
    A: torch.Tensor,
    scales_q: torch.Tensor,
    scales_dq: torch.Tensor,
) -> torch.Tensor:
    qscheme = create_normal_float_scheme(
        num_bits=4,
        device=A.device)

    _, A_dequantized = quantize_with_scheme(
        A,
        qscheme=qscheme,
        scales_q=scales_q,
        scales_dq=scales_dq)
    return A_dequantized


def dimwise_absmax(A: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.max(
        torch.abs(A),
        dim=dim,
        keepdim=True).values


def blockwise_absmax(
    A: torch.Tensor,
    num_bits_0: int,  # of the second-level quantization
    num_bits_1: str,  # of the second-level quantization states
    block_size_0: int,
    block_size_1: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (TODO) Double check this
    if A.dtype != torch.float32:
        raise ValueError(f"Expected float32, but got {A.dtype}")
    if num_bits_1 == "bf16":
        dtype = torch.bfloat16
    elif num_bits_1 == "fp16":
        dtype = torch.float16
    elif num_bits_1 == "fp32":
        dtype = torch.float32
    else:
        raise ValueError

    # Compute the second-level quantization
    scales_0 = A.view(-1, block_size_1, block_size_0)
    scales_1 = dimwise_absmax(scales_0, dim=2)
    # Notice that we use the `.min` as the offset.
    # This guarantees that the the smallest number after
    # quantization will be at least `offset_1`, which is
    # positive because `scales_1` is non-negative.
    offset_1 = scales_1.min()
    scales_2 = scales_1 - offset_1
    scales_3 = dimwise_absmax(scales_2, dim=1)
    # (TODO) Double check this
    scales_3 = (
        scales_3
        .to(dtype=dtype)
        .to(dtype=scales_3.dtype))

    # Reconstruct the first-level quantization scales
    scales_3_ = torch.broadcast_to(scales_3, scales_2.shape)
    # (Unsigned) int8 quantization of the first-level scales
    scales_2_ = quantize_with_scheme_2(
        scales_2,
        scales=scales_3_,
        num_bits=num_bits_0,
        dtype="uint")
    scales_1_ = scales_2_ + offset_1

    # `scales_q` is the `scales` for quantizing `A`
    # `scales_dq` is the `scales` for dequantizing `A`
    scales_q = torch.broadcast_to(scales_1, scales_0.shape)
    scales_dq = torch.broadcast_to(scales_1_, scales_0.shape)
    scales_q = scales_q.reshape(A.shape)
    scales_dq = scales_dq.reshape(A.shape)
    return scales_q, scales_dq


def blockwise_absmax_original(
    A: torch.Tensor,
    block_size_0: int,
    block_size_1: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Compute the second-level quantization
    scales_0 = A.view(-1, block_size_1, block_size_0)
    scales_1 = dimwise_absmax(scales_0, dim=2)
    offset_1 = scales_1.mean()
    scales_2 = scales_1 - offset_1
    scales_3 = dimwise_absmax(scales_2, dim=1)

    # Reconstruct the first-level quantization scales
    scales_3_ = torch.broadcast_to(scales_3, scales_2.shape)
    # FP8 quantization of the first-level scales
    scales_2_ = quantize_with_scheme_2(
        scales_2,
        scales=scales_3_,
        num_bits=8,
        dtype="fp")
    scales_1_ = scales_2_ + offset_1

    # `scales_q` is the `scales` for quantizing `A`
    # `scales_dq` is the `scales` for dequantizing `A`
    scales_q = torch.broadcast_to(scales_1, scales_0.shape)
    scales_dq = torch.broadcast_to(scales_1_, scales_0.shape)
    scales_q = scales_q.reshape(A.shape)
    scales_dq = scales_dq.reshape(A.shape)
    return scales_q, scales_dq


def estimate_storage_from_config(
    A: torch.Tensor,
    qconfig: QuantConfig,
) -> float:
    b = float(qconfig.num_bits)
    b0 = float(qconfig.num_bits_0)
    B0 = float(qconfig.block_size_0)
    B1 = float(qconfig.block_size_1)
    if qconfig.num_bits_1 == "fp32":
        b1 = 32.
    elif qconfig.num_bits_1 in ["bf16", "fp16"]:
        b1 = 16.
    else:
        raise ValueError

    overhead_0 = b0 / B0
    overhead_1 = b1 / (B0 * B1)
    total_bits = b + overhead_0 + overhead_1
    return total_bits * A.numel()


def quantize(
    A: torch.Tensor,
    method: str,
    qconfig: QuantConfig,
) -> torch.Tensor:
    if method not in [
        "blockwise-nf-original",
        "blockwise-nf",
        "blockwise-int",
        "dimwise-nf",
        "dimwise-int"]:
        raise ValueError(f"Unknown quantization method: {method}")

    misc_utils.swarn(
        f"Quantization scheme: "
        f"method={method}, "
        f"qconfig={qconfig}",
        fg="red")

    if method == "blockwise-nf-original":
        if not all([
            qconfig.num_bits == 4,
            qconfig.num_bits_0 == 8,
            qconfig.num_bits_1 == "fp32",
            qconfig.block_size_0 == 64,
            qconfig.block_size_1 == 256]):
            raise ValueError

        scales_q, scales_dq = blockwise_absmax_original(
            A,
            block_size_0=64,
            block_size_1=256)
        return quantize_with_nf4_original(
            A,
            scales_q=scales_q,
            scales_dq=scales_dq)

    if method == "blockwise-nf":
        _, scales_dq = blockwise_absmax(
            A,
            num_bits_0=qconfig.num_bits_0,
            num_bits_1=qconfig.num_bits_1,
            block_size_0=qconfig.block_size_0,
            block_size_1=qconfig.block_size_1)
        return quantize_with_scheme_2(
            A,
            scales=scales_dq,
            num_bits=qconfig.num_bits,
            dtype="nf")

    if method == "blockwise-int":
        _, scales_dq = blockwise_absmax(
            A,
            num_bits_0=qconfig.num_bits_0,
            num_bits_1=qconfig.num_bits_1,
            block_size_0=qconfig.block_size_0,
            block_size_1=qconfig.block_size_1)
        return quantize_with_scheme_2(
            A,
            scales=scales_dq,
            num_bits=qconfig.num_bits,
            dtype="int")

    if method == "dimwise-nf":
        scales = dimwise_absmax(A, dim=0)
        return quantize_with_scheme_2(
            A,
            scales=scales,
            num_bits=qconfig.num_bits,
            dtype="nf")

    if method == "dimwise-int":
        scales = dimwise_absmax(A, dim=0)
        return quantize_with_scheme_2(
            A,
            scales=scales,
            num_bits=qconfig.num_bits,
            dtype="int")

    raise ValueError
