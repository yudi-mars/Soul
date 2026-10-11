"""
Filename:
    PSN.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-13

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Wei Fang et al., "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies", NeurIPS'2023.
    https://github.com/fangwei123456/Parallel-Spiking-Neuron
"""
import math

import torch
import torch.nn as nn

from soul.neuron.base import surrogate_from_config


class ParallelSpikingNode(nn.Module):
    """
    并行脉冲神经元：不保存跨调用的状态，用可学习的 [T, T] 矩阵一次性混合整段序列，
    h = W @ x + b，spike = surrogate_function(h)。输入 [T, N, *] 的时间步数必须等于 time_step。

    配置项：time_step, surrogate
    """
    def __init__(self, config):
        super().__init__()
        self.T = config['time_step']
        self.surrogate_function = surrogate_from_config(config)
        weight = torch.zeros([self.T, self.T])
        bias = torch.zeros([self.T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

    def extra_repr(self):
        return f'T={self.T}'

    def forward(self, x_seq: torch.Tensor):
        if x_seq.shape[0] != self.T:
            raise ValueError(f'{self._get_name()} is built for time_step={self.T}, but got an input with {x_seq.shape[0]} time steps')
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)
