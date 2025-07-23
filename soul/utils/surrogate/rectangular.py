"""
Filename: rectangular.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-06-02
Description:
    The Rectangular surrogate spiking function.

References:
    - Yujie Wu et al., "Direct Training for Spiking Neural Networks: Faster, Larger, Better", AAAI'2019.
"""
import torch

from .base import SurrogateFunctionBase, heaviside

@torch.jit.script
def rectangular_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    temp = (abs(x) < (alpha / 2)) / alpha
    return temp.float() * grad_output, None
 
class rectangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):    
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return rectangular_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)
    
class Rectangular(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangular.apply(x, alpha)
    
    @staticmethod
    def backward(grad_output, x, alpha):
        return rectangular_backward(grad_output, x, alpha)[0]
    
class rectangle(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, vth):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.vth = vth
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x = ctx.saved_tensors[0]
            mask1 = (x.abs() > ctx.vth / 2)
            mask_ = mask1.logical_not()
            grad_x = grad_output * x.masked_fill(mask_, 1. / ctx.vth).masked_fill(mask1, 0.)
        return grad_x, None

class Rectangle(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return rectangle.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return torch.min(torch.max(1. / alpha * x, 0.5), -0.5)