import math
import torch
import jaxtyping
from typing import Tuple, Union, Optional, NamedTuple

PackedDType = torch.uint8
PackedNumBits = torch.iinfo(PackedDType).bits
FloatTensorType = jaxtyping.Float[torch.Tensor, "..."]
UInt8TensorType = jaxtyping.UInt8[torch.Tensor, "..."]
Int32TensorType = jaxtyping.Int32[torch.Tensor, "..."]
BinaryTensorType = jaxtyping.Bool[torch.Tensor, "..."]
PackedBinaryTensorType = Union[UInt8TensorType, Int32TensorType]


# https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
def to_binary(tensor: UInt8TensorType, num_bits: int, legacy: bool = True) -> BinaryTensorType:
    if tensor.dtype != torch.uint8:
        raise TypeError
    if num_bits > 8:
        raise NotImplementedError

    # Explicit casting, and the following code will
    # raise an Error if casting leads to overflow
    bits_max = torch.tensor(
        2 ** num_bits - 1,
        dtype=torch.uint8,
        device=tensor.device)
    if tensor.max() > bits_max:
        raise OverflowError

    if legacy is True:
        # When using `torch.compile`, the `pow` ops
        # requires floating point numbers, but the
        # `bitwise_and` requires integers.
        mask = 2 ** torch.arange(
            num_bits - 1, -1, -1,
            dtype=torch.float32,
            device=tensor.device)
        mask = mask.to(dtype=torch.uint8)
    else:
        # 1. The above casting is not necessary for PyTorch>=2.1
        # 2. We no longer reverse the bits directions
        mask = 2 ** torch.arange(
            num_bits,
            dtype=torch.uint8,
            device=tensor.device)

    return (
        tensor
        .unsqueeze(dim=-1)
        .bitwise_and(mask)
        .ne(0)
        .bool())


def from_binary(tensor: BinaryTensorType, num_bits: int, legacy: bool = True) -> UInt8TensorType:
    if tensor.dtype != torch.bool:
        raise TypeError
    if tensor.shape[-1] != num_bits:
        raise ValueError
    if num_bits > 8:
        raise NotImplementedError

    if legacy is True:
        mask = 2 ** torch.arange(
            num_bits - 1, -1, -1,
            dtype=torch.float32,
            device=tensor.device)
        mask = mask.to(dtype=torch.uint8)
    else:
        mask = 2 ** torch.arange(
            num_bits,
            dtype=torch.uint8,
            device=tensor.device)

    # This casting is somewhat unnecessary.
    tensor = tensor.to(dtype=torch.uint8)
    output = torch.sum(mask * tensor, dim=-1)
    output = output.to(dtype=torch.uint8)
    return output


def pack_bools_into_integers(
    tensor: BinaryTensorType,
    packed_dtype: torch.dtype,
) -> Tuple[PackedBinaryTensorType, int]:
    if tensor.ndim != 1:
        raise ValueError
    if tensor.dtype != torch.bool:
        raise TypeError
    if packed_dtype not in [torch.uint8, torch.int32]:
        raise NotImplementedError

    # number of bits in the packed dtype
    packed_num_bits = torch.iinfo(packed_dtype).bits

    remainder = (
        tensor.shape[-1] %
        packed_num_bits)
    if remainder > 0:
        padding_length = (
            packed_num_bits -
            remainder)
        padding = tensor.new_zeros(padding_length)
        tensor = torch.cat([tensor, padding], dim=-1)
    else:
        padding_length = 0

    # [-1, packed_num_bits]
    tensor = tensor.view(
        int(tensor.shape[-1] / packed_num_bits),
        packed_num_bits)

    # [1, packed_num_bits]
    bits = torch.arange(
        packed_num_bits,
        dtype=packed_dtype,
        device=tensor.device)
    bits = torch.unsqueeze(bits, dim=0)
    packed_tensor = (tensor << bits)
    packed_tensor = torch.sum(packed_tensor, dim=-1)
    packed_tensor = packed_tensor.to(dtype=packed_dtype)
    return packed_tensor, padding_length


def unpack_integers_into_bools(
    packed_tensor: PackedBinaryTensorType,
    padding_length: int,
    packed_dtype: torch.dtype,
) -> BinaryTensorType:
    if packed_tensor.ndim != 1:
        raise ValueError
    if packed_tensor.dtype != packed_dtype:
        raise TypeError
    if packed_dtype not in [torch.uint8, torch.int32]:
        raise NotImplementedError

    # number of bits in the packed dtype
    packed_num_bits = torch.iinfo(packed_dtype).bits

    # [1, packed_num_bits]
    bits = packed_tensor.new_tensor(
        1,
        dtype=packed_dtype)
    bits = bits << torch.arange(
        packed_num_bits,
        dtype=packed_dtype,
        device=packed_tensor.device)
    bits = torch.unsqueeze(
        bits,
        dim=0)
    unpacked_tensor = torch.unsqueeze(
        packed_tensor,
        dim=-1)
    unpacked_tensor = unpacked_tensor & bits
    if packed_dtype == torch.uint8:
        unpacked_tensor = unpacked_tensor > 0
    elif packed_dtype == torch.int32:
        # For signed integers such as int32, the 31st element is the
        # sign bit, so 0b10000000000000000000000000000000 = -2^31
        # The following line of code can be applied to both settings.
        # However, for legacy reasons, we only apply it to int32.
        unpacked_tensor = unpacked_tensor != 0
    else:
        raise NotImplementedError

    unpacked_tensor = unpacked_tensor.to(dtype=torch.bool)
    unpacked_tensor = unpacked_tensor.view(-1)
    if padding_length > 0:
        unpacked_tensor = unpacked_tensor[:-padding_length]
    return unpacked_tensor


# @torch.compile
def pack_integer_tensors(
    tensor: UInt8TensorType,
    num_bits: int,
) -> Tuple[PackedBinaryTensorType, int]:
    # [*tensor.shape, num_bits]
    binary_tensor = to_binary(
        tensor=tensor,
        num_bits=num_bits,
        legacy=True)
    # [tensor.numel() x num_bits]
    binary_tensor = binary_tensor.view(
        tensor.numel() * num_bits)
    binary_tensor = binary_tensor.contiguous()
    # [tensor.numel() x num_bits / 8]
    return pack_bools_into_integers(
        tensor=binary_tensor,
        packed_dtype=PackedDType)


# @torch.compile
def unpack_integer_tensors(
    packed_tensor: PackedBinaryTensorType,
    padding_length: int,
    num_bits: int,
    shape: Tuple[int, ...],
) -> UInt8TensorType:
    packed_size = (
        (math.prod(shape) * num_bits + padding_length) /
        PackedNumBits)
    if packed_tensor.shape != (packed_size,):
        raise ValueError

    # [tensor.numel() x num_bits / 8]
    packed_tensor = packed_tensor.contiguous()
    # [tensor.numel() x num_bits]
    binary_tensor = unpack_integers_into_bools(
        packed_tensor=packed_tensor,
        padding_length=padding_length,
        packed_dtype=PackedDType)
    # [*tensor.shape, num_bits]
    binary_tensor = binary_tensor.view(
        *shape, num_bits)
    return from_binary(
        tensor=binary_tensor,
        num_bits=num_bits,
        legacy=True)

# -----------------------------------------------------------------------------
# Fast (V2)
# -----------------------------------------------------------------------------

PackedDType2 = torch.int32
PackedNumBits2 = torch.iinfo(PackedDType2).bits


class BitsConfig(NamedTuple):
    mask: Int32TensorType
    shifts: Int32TensorType
    num_bits: int
    num_packed: int


def get_bits_config(num_bits: int) -> BitsConfig:
    if num_bits in [2, 4, 8]:
        mask = torch.tensor(
            2 ** num_bits - 1,
            dtype=PackedDType2,
            device="cuda")

        shifts = torch.arange(
            0, PackedNumBits2, num_bits,
            dtype=PackedDType2,
            device="cuda")

        return BitsConfig(
            mask=mask,
            shifts=shifts,
            num_bits=num_bits,
            num_packed=shifts.shape[-1])

    if num_bits == 3:
        mask = torch.tensor(
            1,
            dtype=PackedDType2,
            device="cuda")

        shifts = torch.arange(
            PackedNumBits2,
            dtype=PackedDType2,
            device="cuda")

        return BitsConfig(
            mask=mask,
            shifts=shifts,
            num_bits=num_bits,
            num_packed=shifts.shape[-1])

    raise NotImplementedError


BitsConfigDict = {
    2: get_bits_config(2),
    3: get_bits_config(3),
    4: get_bits_config(4),
    8: get_bits_config(8),
}


def pack_integer_tensors_2(
    tensor: UInt8TensorType,
    num_bits: int,
) -> PackedBinaryTensorType:
    # Two major differences for faster dequantization
    # 1. `reverse=False`
    # 2. `packed_dtype=torch.int32`
    # 3. special implementation for `num_bits=3`
    # 4. does not support padding

    # [*tensor.shape, num_bits]
    binary_tensor = to_binary(
        tensor=tensor,
        num_bits=num_bits,
        legacy=False)

    if num_bits == 3:
        # This makes dequantization less efficient in general,
        # but this makes 3-bit dequantization more efficient.
        # This packs 32 3-bit values into 3 32-bit values, such that
        # i-th position in the first value is the first bit of the i-th
        # 3-bit value, i-th position in the second value is the second
        # bit of the i-th 3-bit value, and so on.
        if binary_tensor.shape[-1] != num_bits:
            raise ValueError
        binary_tensor = binary_tensor.view(-1, PackedNumBits2, num_bits)
        binary_tensor = binary_tensor.transpose(dim0=1, dim1=2)
        binary_tensor = binary_tensor.contiguous()

    # [tensor.numel() x num_bits]
    binary_tensor = binary_tensor.view(
        tensor.numel() * num_bits)
    binary_tensor = binary_tensor.contiguous()
    # [tensor.numel() x num_bits / 32]
    packed_tensor, padding_length = pack_bools_into_integers(
        tensor=binary_tensor,
        packed_dtype=PackedDType2)
    if padding_length != 0:
        raise ValueError
    return packed_tensor


def unpack_integer_tensors_2(
    packed_tensor: PackedBinaryTensorType,
    num_bits: int,
    shape: Tuple[int, ...],
) -> Int32TensorType:
    packed_size = (math.prod(shape) * num_bits) / PackedNumBits2
    if packed_tensor.shape != (packed_size,):
        raise ValueError
    if packed_tensor.dtype != PackedDType2:
        raise TypeError

    bconfig = BitsConfigDict[num_bits]
    if bconfig.num_bits != num_bits:
        raise ValueError

    # [..., num_packed]
    # b=2,4,8: [..., 32 / b], numel = prod(shape)
    # b=3    : [..., 32]    , numel = prod(shape) * b
    unpacked_tensor = packed_tensor.contiguous()
    # explicit unsqueezing + broadcasting, as the
    # compiler might not figure that out
    unpacked_tensor = unpacked_tensor.unsqueeze(dim=-1)
    unpacked_tensor = unpacked_tensor.broadcast_to(
        *packed_tensor.shape,
        bconfig.num_packed)

    if num_bits in [2, 4, 8]:
        # Apply masks and right shift
        # Example 4-bit case,
        # packed_tensor[i] = hhhh gggg ffff eeee dddd cccc bbbb aaaa
        # after the unsqueezing + broadcasting and before the shifts:
        # unpacked_tensor[i, :] = [
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        # ]
        # after the shifts (`.` could be 0/1 because of arithmetic shift)
        # unpacked_tensor[i, :] = [
        #   hhhh gggg ffff eeee dddd cccc bbbb aaaa,
        #   .... hhhh gggg ffff eeee dddd cccc bbbb,
        #   .... .... hhhh gggg ffff eeee dddd cccc,
        #   .... .... .... hhhh gggg ffff eeee dddd,
        #   .... .... .... .... hhhh gggg ffff eeee,
        #   .... .... .... .... .... hhhh gggg ffff,
        #   .... .... .... .... .... .... hhhh gggg,
        #   .... .... .... .... .... .... .... hhhh,
        # ]
        unpacked_tensor = unpacked_tensor >> bconfig.shifts
        # unpacked_tensor[i, :] = [
        #   0000 0000 0000 0000 0000 0000 0000 aaaa,
        #   0000 0000 0000 0000 0000 0000 0000 bbbb,
        #   0000 0000 0000 0000 0000 0000 0000 cccc,
        #   0000 0000 0000 0000 0000 0000 0000 dddd,
        #   0000 0000 0000 0000 0000 0000 0000 eeee,
        #   0000 0000 0000 0000 0000 0000 0000 ffff,
        #   0000 0000 0000 0000 0000 0000 0000 gggg,
        #   0000 0000 0000 0000 0000 0000 0000 hhhh,
        # ]
        unpacked_tensor = unpacked_tensor & bconfig.mask

    elif num_bits == 3:
        # [-1, 32] -> [-1, 3, 32]
        # Example:
        # packed_tensor[3 * i : 4 * i] = [
        #   .... .... .... .... .... .... hgfe dcba,  --> 1st bit
        #   .... .... .... .... .... .... hgfe dcba,  --> 2nd bit
        #   .... .... .... .... .... .... hgfe dcba,  --> 3rd bit
        # ]
        # unpacked_tensor[i, :, :] = [
        #   [
        #       .... .... .... .... .... .... hgfe dcba,  --> 1st bit
        #       .... .... .... .... .... .... hgfe dcba,  --> 1st bit
        #       .... .... .... .... .... .... hgfe dcba,  --> 1st bit
        #       ...,
        #   ],
        #   [
        #       .... .... .... .... .... .... hgfe dcba,  --> 2nd bit
        #       .... .... .... .... .... .... hgfe dcba,  --> 2nd bit
        #       .... .... .... .... .... .... hgfe dcba,  --> 2nd bit
        #       ...,
        #   ],
        #   [
        #       .... .... .... .... .... .... hgfe dcba,  --> 3rd bit
        #       .... .... .... .... .... .... hgfe dcba,  --> 3rd bit
        #       .... .... .... .... .... .... hgfe dcba,  --> 3rd bit
        #       ...,
        #   ]
        # ]
        unpacked_tensor = unpacked_tensor.view(-1, num_bits, bconfig.num_packed)
        # unpacked_bits_0[i, :] = [
        #   0000 0000 0000 0000 0000 0000 0000 000a,  --> 1st bit
        #   0000 0000 0000 0000 0000 0000 0000 000b,  --> 1st bit
        #   0000 0000 0000 0000 0000 0000 0000 000c,  --> 1st bit
        #   ...,
        # ]
        # Note that left shift is the same between logical and arithmetic, and just fill them
        # with zeros. So we don't need to proceed it by more masking.
        unpacked_bits_0 = ((unpacked_tensor[:, 0, :] >> bconfig.shifts) & bconfig.mask) << 0
        # unpacked_bits_1[i, :] = [
        #   0000 0000 0000 0000 0000 0000 0000 00a0,  --> 2nd bit
        #   0000 0000 0000 0000 0000 0000 0000 00b0,  --> 2nd bit
        #   0000 0000 0000 0000 0000 0000 0000 00c0,  --> 2nd bit
        #   ...,
        # ]
        unpacked_bits_1 = ((unpacked_tensor[:, 1, :] >> bconfig.shifts) & bconfig.mask) << 1
        # unpacked_bits_2[i, :] = [
        #   0000 0000 0000 0000 0000 0000 0000 0a00,  --> 3rd bit
        #   0000 0000 0000 0000 0000 0000 0000 0b00,  --> 3rd bit
        #   0000 0000 0000 0000 0000 0000 0000 0c00,  --> 3rd bit
        #   ...,
        # ]
        unpacked_bits_2 = ((unpacked_tensor[:, 2, :] >> bconfig.shifts) & bconfig.mask) << 2
        # [...zyx] = [...00x] OR [...0y0] OR [...z00]
        # unpacked_tensor[i, :] = [
        #   0000 0000 0000 0000 0000 0000 0000 0aaa,
        #   0000 0000 0000 0000 0000 0000 0000 0bbb,
        #   0000 0000 0000 0000 0000 0000 0000 0ccc,
        #   ...,
        # ]
        # [-1, 3, 32] -> [-1, 32]
        unpacked_tensor = unpacked_bits_0 | unpacked_bits_1 | unpacked_bits_2

    else:
        raise ValueError

    return unpacked_tensor.view(shape)

