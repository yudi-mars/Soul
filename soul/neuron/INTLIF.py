"""
Filename:
    INTLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-07

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Xinhao Luo et al., "Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection", ECCV'2024.
    https://github.com/BICLab/SpikeYOLO
"""
import torch

from soul.neuron.base import BaseNode


class INTLIFNode(BaseNode):
    """
    整数值 LIF 神经元：脉冲为量化后的膜电位 surrogate_function(v)（如 Quant: round(clamp(v, 0, 8))），
        充电 v <- v / tau + x，重置总是软重置 v <- v - spike * v_threshold（v_reset 只作为膜电位初值），
        输出 spike / norm。

    默认使用单步模式：整段 [T, N, *] 输入作为一个时间步积分，即各时间步独立量化（整数值训练）；
    设置 step_mode='m' 则沿时间维逐步积分。

    额外配置项：tau, norm
    """
    default_step_mode = 's'

    def __init__(self, config):
        super().__init__(config)
        self.tau = config['tau']
        self.norm = config['norm']

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v / self.tau + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v)

    def neuronal_reset(self, spike: torch.Tensor):
        if self.detach_reset:
            spike = spike.detach()
        self.v = self.v - spike * self.v_threshold

    def single_step_forward(self, x: torch.Tensor):
        return super().single_step_forward(x) / self.norm

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, norm={self.norm}'
