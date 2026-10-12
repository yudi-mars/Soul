"""
Filename:
    DLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2026-10-12

Description:
    implementation for LIF neurons with dendritic bilinear integration for Spiking Neural Networks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from soul.neuron.LIF import LIFNode


class DLIFNode(LIFNode):
    """
    树突 LIF 神经元：胞体为 LIF 动力学（缺省 tau=1.1、不衰减输入、软重置）；树突对输入的双线性整合由
    DendriticBilinear 层完成，只在使用该层的模型（lenet_dlif）中生效，其他模型中等同于该参数下的 LIF。

    配置项同 LIFNode
    """


class DendriticBilinear(nn.Module):
    """
    树突双线性整合：在每个空间位置上把输入通道两两相乘（不含自身平方项），再线性组合，
        y_o = sum_{i != j} W_oij * x_i * x_j + b_o
    输入 [N, in_channels, *]，输出 [N, out_channels, *]，支持任意空间维数。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        mask = (1. - torch.eye(in_channels)).expand(out_channels, in_channels, in_channels).clone()
        self.register_buffer('mask', mask, persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.weight.shape[1] * self.weight.shape[2])
        nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def extra_repr(self):
        return f'in_channels={self.weight.shape[1]}, out_channels={self.weight.shape[0]}'

    def forward(self, x: torch.Tensor):
        x = x.movedim(1, -1)
        outer = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)
        y = F.linear(outer, (self.weight * self.mask).flatten(1), self.bias)
        return y.movedim(-1, 1)
