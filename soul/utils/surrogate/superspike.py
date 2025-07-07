import torch
import math
from .base import SurrogateFunctionBase, heaviside

@torch.jit.script
def atan_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2)) * grad_output, None

class atan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return atan_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

@torch.jit.script
def super_spike_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha * grad_output / torch.pow(torch.abs(x) + 1., 2), None

class super_spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return super_spike_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)
    
class SuperSpike(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return atan.apply(x, alpha)

    @staticmethod
    def backward(grad_output, x, alpha):
        return super_spike_backward(grad_output, x, alpha)[0]
