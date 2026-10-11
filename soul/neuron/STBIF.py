"""
Filename:
    STBIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-15

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Kang You et al., "VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network", CVPR'2025.
    https://github.com/Intelligent-Computing-Research-Group/ViStream
"""
from collections import ChainMap

import torch

from soul.neuron.base import BaseNode


class STBIF(BaseNode):
    """
    ST-BIF（Integrate-and-Fire）神经元：膜电位 v 保存按 q_threshold 归一化后的电荷，从 0.5 开始积分，
    v >= 1 时发放并减 1（相当于对累积输入四舍五入），输出 spike * q_threshold。
    反向传播时脉冲对 v 的梯度由 surrogate_function(v - 1) 给出；v_threshold / v_reset 不起作用。

    额外配置项：q_threshold（量化门限）, level（量化等级）, sym（是否对称量化，缺省 False，目前未使用）,
              detach_reset（缺省 True）
    """
    def __init__(self, config):
        super().__init__(ChainMap(config, {'detach_reset': True}))
        self.q_threshold = float(config['q_threshold'])
        self.level = config['level']
        self.sym = config.get('sym', False)
        self.v = 0.5
        self.set_reset_value('v', 0.5)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float) or self.v.shape != x.shape:
            self.v = torch.full_like(x, 0.5, dtype=torch.float32)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x / self.q_threshold

    def neuronal_fire(self):
        spike = self.surrogate_function(self.v - 1.)
        # the surrogate fires at v > 1, ST-BIF fires at v >= 1: keep the exact spikes and the surrogate gradient
        return spike + ((self.v >= 1.).to(spike) - spike).detach()

    def neuronal_reset(self, spike: torch.Tensor):
        if self.detach_reset:
            spike = spike.detach()
        self.v = self.v - spike

    def single_step_forward(self, x: torch.Tensor):
        return super().single_step_forward(x).to(x.dtype) * self.q_threshold

    def extra_repr(self):
        return f'q_threshold={self.q_threshold}, level={self.level}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'
