"""
Filename: IELIF.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-05-05
Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Xuerui Qiu et al., "Quantized Spike-Driven Transformer", ICLR'2025.
    https://github.com/bollossom/QSD-Transformer
"""
import torch
import torch.nn as nn

class ReLUX(nn.Module):
    def __init__(self, thre=8):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(4)

class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None

class IELIFNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lens = config["lens"]
        self.spike = multispike

    def forward(self, inputs):
        return self.spike.apply(4 * inputs, self.lens) / 4
    