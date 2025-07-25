"""
Filename: quantize.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-05-30
Description:
    specific surrogate function for INTLIF

References:
    - Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025.
"""
import torch
from .base import SurrogateFunctionBase

class quant4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, min_value=0, max_value=4): 
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class Quant4(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
    @staticmethod
    def spiking_function(x, alpha):
         return quant4.apply(x) / 4
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()


class quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, min_value=0, max_value=8): 
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None

class Quant(SurrogateFunctionBase):
    def __init__(self, alpha=4.0, spiking=True):
        super().__init__(alpha, spiking)
    @staticmethod
    def spiking_function(x, alpha):
         return quant.apply(x)
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha).sigmoid()
    
