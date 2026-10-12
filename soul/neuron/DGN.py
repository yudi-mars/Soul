"""
Filename:
    DGN.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2026-10-12

Description:
    implementation for Dynamic Gated Neuron for Spiking Neural Networks.

References:
    - Qianyi Bai et al., "A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks", ICLR'2026.
"""
import math

import torch
import torch.nn as nn

from soul.neuron.base import BaseNode


class DGNNode(BaseNode):
    """
    Dynamic Gated Neuron：膜电位的衰减系数由动态电导门控，
        D_t   = exp(-1 / tau_s) * D_{t-1} + x_t        （突触电流迹，电流、电导两路输入各维护一条）
        rho_t = sigmoid(1 - g_l - D^g_t)               （D^g 为电导输入的迹）
        v_t   = rho_t * v_{t-1} + D^c_t                （D^c 为电流输入的迹），随后发放并重置（缺省软重置）

    forward(x, g=None)：传入电导输入 g 时 x 为电流输入，即完整的 DGN（需要分别计算电流与电导的双路径模型）；
    只传 x 时按 s-DGN 运行：所有突触共享平衡电位 E，x 作为电导输入，电流为 E * D^g，E 可学习。

    额外配置项：g_l（漏电导）, tau_s（突触时间常数，None 表示不做突触滤波）, init_equilibrium（E 的初值，缺省 1）
    """
    def __init__(self, config):
        super().__init__(config)
        self.g_l = config['g_l']
        self.tau_s = config['tau_s']
        self.equilibrium = nn.Parameter(torch.tensor(float(config.get('init_equilibrium', 1.))))
        self.register_memory('d', 0.)
        self.register_memory('d_c', 0.)

    @property
    def synaptic_decay(self):
        return 0. if self.tau_s is None else math.exp(-1. / self.tau_s)

    def neuronal_charge(self, x: torch.Tensor, g: torch.Tensor = None):
        decay = self.synaptic_decay
        if g is None:
            self.d = self.d * decay + x
            current = self.equilibrium * self.d
        else:
            self.d = self.d * decay + g
            self.d_c = self.d_c * decay + x
            current = self.d_c
        self.v = torch.sigmoid(1. - self.g_l - self.d) * self.v + current

    def single_step_forward(self, x: torch.Tensor, g: torch.Tensor = None):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x, g)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def extra_repr(self):
        return super().extra_repr() + f', g_l={self.g_l}, tau_s={self.tau_s}'
