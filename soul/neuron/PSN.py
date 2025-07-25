"""
Filename: PSN.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-05-13
Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Wei Fang et al., "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies", NeurIPs'2023.
    https://github.com/fangwei123456/Parallel-Spiking-Neuron
"""
import math
import torch
import torch.nn as nn

class ParallelSpikingNode(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.T = config['T']
        self.surrogate_function = config['surrogate_function']
        weight = torch.zeros([self.T, self.T])
        bias = torch.zeros([self.T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)
