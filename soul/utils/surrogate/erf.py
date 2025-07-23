"""
Filename: erf.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-06-01
Description:
    The Gaussian error (erf) surrogate spiking function.

References:
    - Esser S. K. et al., "Backpropagation for energy-efficient neuromorphic computing", NeurIPs'2015.
"""
import math
import torch
from .base import SurrogateFunctionBase, heaviside

@torch.jit.script
def erf_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    return grad_output * (- (x * alpha).pow_(2)).exp_() * (alpha / math.sqrt(math.pi)), None

class erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return erf_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)
    
class Erf(SurrogateFunctionBase):
    '''
    Gussian error function
    '''
    def __init__(self, alpha=2.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return erf.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        return torch.erfc_(-alpha * x) / 2.

    @staticmethod
    def backward(grad_output, x, alpha):
        return erf_backward(grad_output, x, alpha)[0]