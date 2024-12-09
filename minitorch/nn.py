from typing import Tuple, Optional, TypeVar, Any

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

from numba import njit as _njit

from .tensor_data import (
    index_to_position,
    to_index,
)

Fn = TypeVar("Fn")

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to JIT compile functions with `nopython=True`."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height / kh
    new_width = width / kw
    input = input.contiguous()
    input = input.view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)

    return input, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies a 2D average pooling operation to the input tensor.

    Args:
    ----
        input (Tensor):
            The input tensor of shape `(batch_size, channels, height, width)`,
            where `batch_size` is the number of samples, `channels` is the
            number of feature maps, and `height` and `width` are the spatial dimensions.

        kernel (Tuple[int, int]):
            A tuple of two integers `(kernel_height, kernel_width)` specifying
            the height and width of the pooling kernel.

    Returns:
    -------
        Tensor:
            The output tensor after applying average pooling, with shape
            `(batch_size, channels, out_height, out_width)`. The dimensions
            `out_height` and `out_width` are determined by the input size and
            kernel size.

    Raises:
    ------
        ValueError:
            If the kernel dimensions are larger than the corresponding dimensions of the input tensor.

    """
    tensor_to_reduce, _, _ = tile(input, kernel=kernel)
    output = (
        tensor_to_reduce.mean(dim=len(tensor_to_reduce.shape) - 1)
        .contiguous()
        .view(*tensor_to_reduce.shape[:-1])
    )

    return output


def argmax(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """Return a one-hot tensor to indicate the argmax"""
    return tensor == tensor.f.max_reduce(tensor, dim)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Call forward for Max"""
        # ctx.save_for_backward(dim)
        dim_int = int(dim.item())
        ctx.save_for_backward(t1, dim_int)
        return t1.f.max_reduce(t1, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Call backward for Max"""
        (t1, dim) = ctx.saved_tensors
        t1_argmax = argmax(t1, dim=dim)
        return t1_argmax * grad_output, 0.0


def max(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """Function max"""
    if dim is not None:
        return Max.apply(tensor, tensor._ensure_tensor(dim))
    else:
        return Max.apply(
            tensor.contiguous().view(tensor.size), tensor._ensure_tensor(0)
        )


def softmax(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """Softmax for tensor"""
    tensor = tensor.exp()
    tensor /= tensor.sum(dim=dim)
    return tensor


def logsoftmax(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """LogSoftmax for tensor"""
    tensor = softmax(tensor, dim=dim)
    tensor = tensor.log()
    return tensor


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies a 2D average pooling operation to the input tensor.

    Args:
    ----
        input (Tensor):
            The input tensor of shape `(batch_size, channels, height, width)`,
            where `batch_size` is the number of samples, `channels` is the
            number of feature maps, and `height` and `width` are the spatial dimensions.

        kernel (Tuple[int, int]):
            A tuple of two integers `(kernel_height, kernel_width)` specifying
            the height and width of the pooling kernel.

    Returns:
    -------
        Tensor:
            The output tensor after applying average pooling, with shape
            `(batch_size, channels, out_height, out_width)`. The dimensions
            `out_height` and `out_width` are determined by the input size and
            kernel size.

    Raises:
    ------
        ValueError:
            If the kernel dimensions are larger than the corresponding dimensions of the input tensor.

    """
    tensor_to_reduce, _, _ = tile(input, kernel=kernel)
    output = (
        max(tensor_to_reduce, dim=len(tensor_to_reduce.shape) - 1)
        .contiguous()
        .view(*tensor_to_reduce.shape[:-1])
    )

    return output


def dropout(tensor: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Drop out at possibility `p`, skip when `ignore`,"""
    if ignore:
        return tensor
    p_tensor = rand(tensor.shape)
    return tensor * (p_tensor > p)


if __name__ == "__main__":
    a = tensor([[1, 2, 3], [4, 5, 6]])
    b = argmax(a)
