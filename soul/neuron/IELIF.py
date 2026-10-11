"""
Filename:
    IELIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-05

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Xuerui Qiu et al., "Quantized Spike-Driven Transformer", ICLR'2025.
    https://github.com/bollossom/QSD-Transformer
"""
import torch
import torch.nn as nn


class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(torch.clamp(input, 0, 4) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class IELIFNode(nn.Module):
    """
    整数值脉冲神经元（无状态）：spike = round(clamp(4x, 0, 4)) / 4，
    反向传播时只在 0 < 4x < lens 范围内传递梯度。

    配置项：lens
    """
    def __init__(self, config):
        super().__init__()
        self.lens = config['lens']

    def extra_repr(self):
        return f'lens={self.lens}'

    def forward(self, inputs: torch.Tensor):
        return multispike.apply(4 * inputs, self.lens) / 4
