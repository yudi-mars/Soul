"""
Filename: RPLIF.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-11-10
Description:
    implementation for Refractory Period LIF.

References:
    - Li, Yang, et al., "Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics", MM'2025.
    https://arxiv.org/pdf/2509.17769
"""
from abc import abstractmethod
import torch
from soul.neuron.LIF import BaseNode


class RPLIFNode(BaseNode):
    """
    Refractory-Period Leaky Integrate-and-Fire (RPLIF) Neuron

    特点：
    - 在 LIF 基础上增加动态阈值 self.v_th
    - 每次发放后： v_th <- v_th * alpha
    - 若未发放且 v_th > v_init_th：v_th <- v_init_th （阈值恢复）
    """

    def __init__(self, config):
        # 基本参数继承自 LIFNode
        v_threshold = config['v_threshold']        # 初始阈值 V_init_th
        v_reset = config['v_reset']
        detach_reset = config['detach_reset']
        backend = 'torch'
        step_mode = 'm'
        store_v_seq = config.get('store_v_seq', False)
        surrogate_function = config['surrogate_function']

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        # 额外超参数
        self.tau = config['tau']
        self.decay_input = config.get('decay_input', True)
        self.alpha = config.get('alpha', 1.2)  # 阈值放大系数 α > 1
        self.v_init_th = torch.tensor(v_threshold, dtype=torch.float32)  # 初始阈值记忆
        self.register_memory('v_th', self.v_init_th.clone())             # 动态阈值

    def neuronal_charge(self, x: torch.Tensor):
        # 与普通 LIF 相同的膜电位更新规则
        if self.decay_input:
            self.v = self.v + (self.v_reset - self.v + x) / self.tau
        else:
            self.v = self.v + (self.v_reset - self.v) / self.tau + x

    def neuronal_fire(self):
        # 使用动态阈值 self.v_th 而非固定 self.v_threshold
        return self.surrogate_function(self.v - self.v_th)

    def neuronal_reset(self, spike):
        # 发放后膜电位复位（与 LIF 相同）
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_th)
        else:
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

        # 阈值动态调整部分（RPLIF 的核心差异）
        with torch.no_grad():
            fired_mask = (spike > 0)
            self.v_th = torch.where(
                fired_mask, self.v_th * self.alpha, self.v_th
            )
            # 未发放但阈值大于初始值时恢复
            self.v_th = torch.where(
                (~fired_mask) & (self.v_th > self.v_init_th),
                torch.full_like(self.v_th, self.v_init_th),
                self.v_th
            )

    def reset(self):
        """清空状态，用于重新运行 forward"""
        super().reset()
        self.v_th = self.v_init_th.clone()

    def extra_repr(self):
        base = super().extra_repr()
        return base + f', tau={self.tau}, alpha={self.alpha}, v_init_th={float(self.v_init_th)}'
       