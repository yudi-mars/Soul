"""
Filename:
    POLARA.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2026-10-12

Description:
    implementation for polarization-aware spiking neurons for Spiking Neural Networks.

References:
    - Matthew Lai and Longbing Cao, "Stabilizing Spiking Neurons Through Biologically Inspired Polarization", AAAI'2026.
"""
import math

import torch

from soul.neuron.base import BaseNode


class POLARANode(BaseNode):
    """
    POLARA 神经元：用时间核函数描述去极化、复极化与超极化，没有发放后的重置。第 t 步（t 从 0 计，i 遍历 t..t+window-1）：
        衰减：      u = v * (1 - A * sum_i exp(-alpha * i) + epsilon)
        随机放大：  u < v_threshold 时 u <- u * (1 + |noise * U(0, 1)|)
        积分：      u <- u + x
        发放：      spike = surrogate(u - theta) * sigmoid(u * O(u - theta))，O(d) = min(d * (1 + d * exp(-d)), o_max)
        不应期抑制：v = u * (1 + sum_i sum_j D_j * exp(-(i - mu_j)^2 / (2 * sigma_j^2)))
        自适应阈值：theta <- theta * (1 + sum_i E * exp(-eta * i))，theta 从 v_threshold 开始
    发放的脉冲取 0 或 [0.5, 1) 内的连续值。

    额外配置项：A, alpha（衰减核）, window（时间窗长度，缺省 5）, noise（随机放大幅度，缺省 0.1，0 表示不加噪声）
    """
    epsilon = 0.1
    o_max = 2.
    refractory_kernel = ((-0.2, 1.5, 0.7), (-0.1, 4., 1.5), (-0.1, 6., 4.))  # (D_j, mu_j, sigma_j)
    threshold_kernel = (0.001, 0.5)  # (E, eta)

    def __init__(self, config):
        super().__init__(config)
        self.A = config['A']
        self.alpha = config['alpha']
        self.window = config.get('window', 5)
        self.noise = config.get('noise', 0.1)
        self.register_memory('t', 0)
        self.register_memory('v_th', None)

    def _lags(self):
        return range(self.t, self.t + self.window)

    def v_float_to_tensor(self, x: torch.Tensor):
        super().v_float_to_tensor(x)
        if self.v_th is None:
            self.v_th = self.v_threshold

    def neuronal_charge(self, x: torch.Tensor):
        u = self.v * (1. - self.A * sum(math.exp(-self.alpha * i) for i in self._lags()) + self.epsilon)
        if self.noise:
            amplified = u * (1. + torch.abs(self.noise * torch.rand_like(u)))
            u = torch.where(u - self.v_threshold < 0, amplified, u)
        self.v = u + x

    def neuronal_fire(self):
        delta = self.v - self.v_th
        overshoot = torch.clamp(delta * (1. + delta * torch.exp(-torch.clamp(delta, min=-10.))), max=self.o_max)
        return self.surrogate_function(delta) * torch.sigmoid(self.v * overshoot)

    def neuronal_reset(self, spike: torch.Tensor):
        lags = self._lags()
        refractory = 1. + sum(d * math.exp(-(i - mu) ** 2 / (2 * sigma ** 2))
                              for i in lags for d, mu, sigma in self.refractory_kernel)
        gain, rate = self.threshold_kernel
        self.v = self.v * refractory
        self.v_th = self.v_th * (1. + sum(gain * math.exp(-rate * i) for i in lags))
        self.t = self.t + 1

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, A={self.A}, alpha={self.alpha}, window={self.window}, noise={self.noise}, step_mode={self.step_mode}, backend={self.backend}'
