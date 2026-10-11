"""
Filename:
    PLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-11

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Wei Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks", ICCV'2021.
    https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
"""
import math

import torch
import torch.nn as nn

from soul.neuron.base import BaseNode


class ParametricLIFNode(BaseNode):
    """
    膜时间常数可学习的 LIF 神经元，1 / tau = sigmoid(w)：
        decay_input=True:  v <- v + (x - (v - v_reset)) * sigmoid(w)
        decay_input=False: v <- v - (v - v_reset) * sigmoid(w) + x

    额外配置项：init_tau（tau 的初值，> 1）, decay_input（缺省 True）
    """
    def __init__(self, config):
        super().__init__(config)
        init_tau = config['init_tau']
        assert isinstance(init_tau, float) and init_tau > 1.
        self.decay_input = config.get('decay_input', True)
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid().item()
        return super().extra_repr() + f', tau={tau:.4f}, decay_input={self.decay_input}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x
