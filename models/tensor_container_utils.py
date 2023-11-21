import torch
import optree
import dataclasses
import torch._dynamo.config
from torch._ops import OpOverload
from torch._ops import ops as torch_ops
from torch.utils import _pytree as pytree
from torch.testing._internal import common_subclass
from torch.overrides import enable_reentrant_dispatch
from typing_extensions import Self
from typing import Tuple, Dict, Any, Optional, Union, Type


# When we use mixed-precision quantization, we do need to
# compile a reasonably large number of times, unfortunately.
torch._dynamo.config.cache_size_limit = 1000
COPY_OP = torch_ops.aten.copy_
CAST_OP = torch_ops.aten._to_copy
DETACH_OP = torch_ops.aten.detach
TRANSPOSE_OP = torch_ops.aten.t
SUPPORTED_TWO_TENSOR_OPS = [
    torch_ops.aten.mm,
    # ops used in LPQ
    torch_ops.aten.add,
    torch_ops.aten.sub,
    torch_ops.aten.rsub,
]
SUPPORTED_THREE_TENSOR_OPS = [
    torch_ops.aten.addmm,
]


@dataclasses.dataclass
class TensorContainer(object):

    @classmethod
    def from_tensor(cls, *args, **kwargs) -> Self:
        raise NotImplementedError

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, state_dict: Dict) -> Self:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        raise NotImplementedError


# `__torch_dispatch__`: https://dev-discuss.pytorch.org/t/what-and-why-is-torch-dispatch/557
# Example 1: https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_subclass.py
# Example 2: https://github.com/pytorch/pytorch/blob/main/test/test_subclass.py
# Example 3: https://github.com/albanD/subclass_zoo/blob/main/quantized_tensor.py
# Handling `torch.nn.Parameter`: https://github.com/pytorch/pytorch/issues/77265
# Connection to `__torch_function__`: https://github.com/albanD/subclass_zoo/issues/46
class QuantizedTensor(common_subclass.WrapperTensor):

    @classmethod
    def get_wrapper_properties(
        cls,
        container: TensorContainer,
        tensor_like: torch.Tensor,
        transpose: bool,
        compute_dtype: Optional[torch.dtype],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if tensor_like.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {tensor_like.shape}.")

        # If `tensor_like` is a `QuantizedTensor`:
        # (1) if `tensor_like._transpose` is False, we use `transpose` as is.
        # (2) if `tensor_like._transpose` is True, we flip `transpose`.
        if isinstance(tensor_like, QuantizedTensor):
            transpose_size = tensor_like._transpose ^ transpose
        else:
            transpose_size = transpose

        if isinstance(tensor_like, QuantizedTensor):
            if (tensor_like._compute_dtype is not None and
                             compute_dtype is None):
                # When `tensor_like` is `QuantizedTensor`,
                # the perceived `dtype` could be different
                # from actual `dtype`. Hence, we need to be
                # explicit about `compute_dtype`.
                raise ValueError

        if not isinstance(transpose_size, bool):
            raise TypeError

        # `tensor_like` and `kwargs` are used to determine
        # how the underlying tensor is perceived by the user
        kwargs = {}

        if transpose_size is True:
            # We do not use `tensor_like.t()` because `tensor_like`
            # itself could be a `QuantizedTensor`, and calling `.t()`
            # would unnecessarily rematerialize the tensor.
            kwargs["size"] = tensor_like.size()[::-1]

        if compute_dtype is not None:
            # This feature is primarily used in AMP. Technically, this
            # is not particularly efficient as we could have cast the
            # data inside the container. But this is simpler for now.
            kwargs["dtype"] = compute_dtype

        return tensor_like, kwargs

    def __init__(
        self,
        container: TensorContainer,
        tensor_like: torch.Tensor,
        transpose: bool,
        compute_dtype: Optional[torch.dtype],
    ) -> None:
        if not isinstance(container, TensorContainer):
            raise TypeError
        self._container = container
        self._transpose = transpose
        self._compute_dtype = compute_dtype

    def __repr__(self) -> str:
        return super().__repr__(tensor_contents=str(self._container))

    # We disable torch function here to avoid any unwanted wrapping of the output
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(
        cls,
        func: OpOverload,
        types: Tuple[Type, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Union[torch.Tensor, Self]:

        # We will apply the compiled function to the combination of
        # (1) dequantization
        # (2) the PyTorch op
        # (3) possibly the `transpose` op.
        if func.overloadpacket in SUPPORTED_TWO_TENSOR_OPS:
            if len(args) != 2:
                raise NotImplementedError
            return apply_three_tensor_op(
                func=func,
                tensor_1=args[0],
                tensor_2=args[1],
                tensor_3=None,
                kwargs=kwargs)

        if func.overloadpacket in SUPPORTED_THREE_TENSOR_OPS:
            if len(args) != 3:
                raise NotImplementedError
            return apply_three_tensor_op(
                func=func,
                tensor_1=args[0],
                tensor_2=args[1],
                tensor_3=args[2],
                kwargs=kwargs)

        # Specifically handle `detach()` since `torch.nn.Parameter`
        # calls `detach` under the hood.
        if func.overloadpacket is DETACH_OP:
            if len(args) != 1 or len(kwargs) != 0:
                raise NotImplementedError
            if type(args[0]) is not QuantizedTensor:
                raise NotImplementedError

            tensor = args[0]
            container = tensor._container
            state_dict_new = pytree.tree_map_only(
                torch.Tensor,
                lambda data: func(data, **kwargs),
                container.to_dict())
            container_new = container.from_dict(state_dict_new)
            return type(tensor)(
                container=container_new,
                tensor_like=tensor,
                transpose=tensor._transpose,
                compute_dtype=tensor._compute_dtype)

        # We will "record" the `transpose` op in a new `QuantizedTensor`
        # object without materializing and transposing the underlying tensor.
        # This is primarily used in `torch.nn.functional.linear`, which does
        # a combination of `transpose` and `mm`.
        if func.overloadpacket is TRANSPOSE_OP:
            if len(args) != 1 or len(kwargs) != 0:
                raise NotImplementedError
            if type(args[0]) is not QuantizedTensor:
                raise NotImplementedError

            tensor = args[0]
            return type(tensor)(
                container=tensor._container,
                tensor_like=tensor,
                transpose=(not tensor._transpose),
                compute_dtype=tensor._compute_dtype)

        # Similarly, we will "record" the `compute_dtype` op.
        if func.overloadpacket is CAST_OP:
            if len(args) != 1:
                raise NotImplementedError
            if kwargs.keys() != {"dtype"}:
                raise NotImplementedError
            if type(args[0]) is not QuantizedTensor:
                raise NotImplementedError

            tensor = args[0]
            return type(tensor)(
                container=tensor._container,
                tensor_like=tensor,
                transpose=tensor._transpose,
                compute_dtype=kwargs["dtype"])

        # This is primarily used in loading the model.
        if func.overloadpacket is COPY_OP:
            if len(args) != 2:
                raise NotImplementedError

            tensor_1, tensor_2 = args
            # Only supporting copying between `QuantizedTensor`
            if not all([
                type(tensor_1) is QuantizedTensor,
                type(tensor_2) is QuantizedTensor,
                tensor_1._transpose == tensor_2._transpose,
                tensor_1._compute_dtype == tensor_2._compute_dtype]):
                raise NotImplementedError

            container_1 = tensor_1._container
            container_2 = tensor_2._container
            # Copying `container_2` into `container_1`
            state_dict_new = optree.tree_map(
                lambda maybe_tensor_1, maybe_tensor_2: _maybe_apply_or_assert(
                    func=func,
                    maybe_tensor_1=maybe_tensor_1,
                    maybe_tensor_2=maybe_tensor_2,
                    kwargs=kwargs),
                container_1.to_dict(),
                container_2.to_dict())
            # Technically, because this is an in-place operation, we can directly return
            # `tensor_1` as its underlying data has been in-place copied. But we will
            # still re-wrap the data into a new container for consistency.
            container_new = container_1.from_dict(state_dict_new)
            return type(tensor_1)(
                container=container_new,
                tensor_like=tensor_1,
                transpose=tensor_1._transpose,
                compute_dtype=tensor_1._compute_dtype)

        func_name = f"{func.__module__}.{func.__name__}"
        raise NotImplementedError(f"Unsupported ops: {func_name} ({func} | {func.overloadpacket})")


def apply_three_tensor_op(
    func: OpOverload,
    tensor_1: Union[torch.Tensor, QuantizedTensor],
    tensor_2: Union[torch.Tensor, QuantizedTensor],
    tensor_3: Union[torch.Tensor, QuantizedTensor, None],
    kwargs: Dict[str, Any],
) -> torch.Tensor:

    # We do all of these seemingly unnecessary maneuvers so that
    # all the inputs to the compiled function are either tensors
    # of trees of tensors. This is because `torch.compile` does not
    # recognize `dataclass` nor custom `torch.Tensor`.
    def _maybe_unwrap_tensor(
        tensor: Union[torch.Tensor, QuantizedTensor],
    ) -> Tuple[Union[torch.Tensor, Dict[str, Any]],
               Optional[Type[TensorContainer]],
               Optional[bool],
               Optional[torch.dtype]]:

        if type(tensor) is not QuantizedTensor:
            return tensor, None, None, None
        return (
            tensor._container.to_dict(),
            type(tensor._container),
            tensor._transpose,
            tensor._compute_dtype)

    maybe_tensor_1, maybe_class_1, maybe_transpose_1, maybe_compute_dtype_1 = _maybe_unwrap_tensor(tensor_1)
    maybe_tensor_2, maybe_class_2, maybe_transpose_2, maybe_compute_dtype_2 = _maybe_unwrap_tensor(tensor_2)
    if tensor_3 is not None:
        maybe_tensor_3, maybe_class_3, maybe_transpose_3, maybe_compute_dtype_3 = _maybe_unwrap_tensor(tensor_3)
    else:
        maybe_tensor_3        = None
        maybe_class_3         = None
        maybe_transpose_3     = None
        maybe_compute_dtype_3 = None

    # https://github.com/albanD/subclass_zoo/issues/47
    with enable_reentrant_dispatch():
        return _compiled_three_tensor_op(
            func=func,
            maybe_tensor_1=maybe_tensor_1,
            maybe_tensor_2=maybe_tensor_2,
            maybe_tensor_3=maybe_tensor_3,
            maybe_class_1=maybe_class_1,
            maybe_class_2=maybe_class_2,
            maybe_class_3=maybe_class_3,
            maybe_transpose_1=maybe_transpose_1,
            maybe_transpose_2=maybe_transpose_2,
            maybe_transpose_3=maybe_transpose_3,
            maybe_compute_dtype_1=maybe_compute_dtype_1,
            maybe_compute_dtype_2=maybe_compute_dtype_2,
            maybe_compute_dtype_3=maybe_compute_dtype_3,
            kwargs=kwargs)


@torch.compile(fullgraph=True)
def _compiled_three_tensor_op(
    func: OpOverload,
    maybe_tensor_1: Union[torch.Tensor, torch.nn.Parameter, Dict[str, Any]],
    maybe_tensor_2: Union[torch.Tensor, torch.nn.Parameter, Dict[str, Any]],
    maybe_tensor_3: Union[torch.Tensor, torch.nn.Parameter, Dict[str, Any], None],
    maybe_class_1: Optional[Type[TensorContainer]],
    maybe_class_2: Optional[Type[TensorContainer]],
    maybe_class_3: Optional[Type[TensorContainer]],
    maybe_transpose_1: Optional[bool],
    maybe_transpose_2: Optional[bool],
    maybe_transpose_3: Optional[bool],
    maybe_compute_dtype_1: Optional[torch.dtype],
    maybe_compute_dtype_2: Optional[torch.dtype],
    maybe_compute_dtype_3: Optional[torch.dtype],
    kwargs: Dict[str, Any],
) -> torch.Tensor:

    # For some reason, PyTorch's compiler does not work well with typing.
    def _maybe_materialize_tensor(
        maybe_tensor,         # Union[torch.Tensor, torch.nn.Parameter, Dict[str, Any]],
        maybe_class,          # Optional[Type[TensorContainer]],
        maybe_transpose,      # Optional[bool],
        maybe_compute_dtype,  # Optional[torch.dtype],
    ) -> torch.Tensor:
        if maybe_class is None:
            if type(maybe_tensor) not in [torch.Tensor, torch.nn.Parameter]:
                raise TypeError
            if maybe_class is not None:
                raise ValueError
            if maybe_transpose is not None:
                raise ValueError
            if maybe_compute_dtype is not None:
                raise ValueError
            return maybe_tensor

        if type(maybe_tensor) is not dict:
            raise TypeError
        if maybe_class is None:
            raise ValueError
        if not isinstance(maybe_transpose, bool):
            raise TypeError

        container = maybe_class.from_dict(maybe_tensor)
        tensor = container.to_tensor()
        if maybe_transpose is True:
            tensor = tensor.t()
        if maybe_compute_dtype is not None:
            tensor = tensor.to(dtype=maybe_compute_dtype)
        return tensor

    tensor_1 = _maybe_materialize_tensor(
        maybe_tensor=maybe_tensor_1,
        maybe_class=maybe_class_1,
        maybe_transpose=maybe_transpose_1,
        maybe_compute_dtype=maybe_compute_dtype_1)
    tensor_2 = _maybe_materialize_tensor(
        maybe_tensor=maybe_tensor_2,
        maybe_class=maybe_class_2,
        maybe_transpose=maybe_transpose_2,
        maybe_compute_dtype=maybe_compute_dtype_2)
    if maybe_tensor_3 is None:
        return func(tensor_1, tensor_2, **kwargs)

    tensor_3 = _maybe_materialize_tensor(
        maybe_tensor=maybe_tensor_3,
        maybe_class=maybe_class_3,
        maybe_transpose=maybe_transpose_3,
        maybe_compute_dtype=maybe_compute_dtype_3)
    return func(tensor_1, tensor_2, tensor_3, **kwargs)


def _maybe_apply_or_assert(
    func: OpOverload,
    maybe_tensor_1: Union[torch.Tensor, torch.nn.Parameter, Any],
    maybe_tensor_2: Union[torch.Tensor, torch.nn.Parameter, Any],
    kwargs: Dict[str, Any],
) -> Union[torch.Tensor, torch.nn.Parameter, Any]:

    if type(maybe_tensor_1) is not type(maybe_tensor_2):
        raise TypeError

    if type(maybe_tensor_1) is torch.nn.Parameter:
        raise TypeError

    if type(maybe_tensor_1) is torch.Tensor:
        return func(maybe_tensor_1, maybe_tensor_2, **kwargs)

    if (torch.is_tensor(maybe_tensor_1) or
        torch.is_tensor(maybe_tensor_2)):
        raise RuntimeError

    # For non-`Tensor` data, we will assert that they are the same.
    if maybe_tensor_1 == maybe_tensor_2:
        # Since they are the same, we just return the first one
        return maybe_tensor_1

    raise ValueError