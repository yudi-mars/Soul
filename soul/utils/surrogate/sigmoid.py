import torch

from .base import SurrogateFunctionBase, heaviside

@torch.jit.script
def sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None


class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)


class Sigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return sigmoid.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha):
        return sigmoid_backward(grad_output, x, alpha)[0]
    
@torch.jit.script
def fast_sigmoid_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    sgax = (x * alpha).sigmoid_()
    return grad_output * (1. - sgax) * sgax * alpha, None


class fast_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return fast_sigmoid_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)
    
class FastSigmoid(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return fast_sigmoid.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (x * alpha).sigmoid()

    @staticmethod
    def backward(grad_output, x, alpha):
        return fast_sigmoid_backward(grad_output, x, alpha)[0]