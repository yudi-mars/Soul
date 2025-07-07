import torch
from .base import SurrogateFunctionBase

class ternary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, alpha):
        ctx.min = -1
        ctx.max = 1
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=-1, max=1))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class Ternary(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
    @staticmethod
    def spiking_function(x, alpha):
         return ternary.apply(x, alpha)
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()