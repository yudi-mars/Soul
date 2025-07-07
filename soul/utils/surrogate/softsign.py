import torch
import torch.nn.functional as F
from .base import SurrogateFunctionBase, heaviside

@torch.jit.script
def soft_sign_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output / (2 * alpha * (1 / alpha + x.abs()).pow_(2)), None

class soft_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return soft_sign_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class SoftSign(SurrogateFunctionBase):
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be lager than 0'

    @staticmethod
    def spiking_function(x, alpha):
        return soft_sign.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return (F.softsign(x * alpha) + 1.) / 2.

    @staticmethod
    def backward(grad_output, x, alpha):
        return soft_sign_backward(grad_output, x, alpha)[0]