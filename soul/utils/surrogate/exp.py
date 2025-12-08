'''
Filename: exp.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-06-02
Description:
    The piecewise exponential surrogate spiking function.

References:
    - Shrestha, S. B. et al., "Slayer: Spike layer error reassignment in time", NeurIPS'2018.
'''
import torch
from .base import SurrogateFunctionBase, HeavisideFunction

heaviside = HeavisideFunction()

@torch.jit.script
def piecewise_exp_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return alpha / 2 * (- alpha * x.abs()).exp_() * grad_output, None


class piecewise_exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_exp_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class PieceWiseExp(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_exp.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask_nonnegative = heaviside(x)
        mask_sign = mask_nonnegative * 2. - 1.
        exp_x = (mask_sign * x * -alpha).exp_() / 2.
        return mask_nonnegative - exp_x * mask_sign

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_exp_backward(grad_output, x, alpha)[0]