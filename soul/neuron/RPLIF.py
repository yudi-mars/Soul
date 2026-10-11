"""
Filename:
    RPLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-11-10

Description:
    implementation for Refractory Period LIF.

References:
    - Li, Yang, et al., "Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics", MM'2025.
    https://arxiv.org/pdf/2509.17769
"""
import torch

from soul.neuron.LIF import LIFNode


class RPLIFNode(LIFNode):
    """
    Refractory-Period LIF 神经元：在 LIF 的基础上使用逐元素的动态阈值 v_th 发放与软重置，
        - 发放后 v_th <- v_th * alpha（alpha > 1），模拟不应期；
        - 未发放且 v_th > v_threshold 时恢复为 v_threshold。

    额外配置项：alpha（缺省 1.2）
    """
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.get('alpha', 1.2)
        self.register_memory('v_th', None)

    def v_float_to_tensor(self, x: torch.Tensor):
        super().v_float_to_tensor(x)
        if self.v_th is None:
            self.v_th = torch.full_like(x.data, self.v_threshold)

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_th)

    def neuronal_reset(self, spike: torch.Tensor):
        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_th
        else:
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

        with torch.no_grad():
            fired_mask = spike > 0
            self.v_th = torch.where(fired_mask, self.v_th * self.alpha, self.v_th)
            self.v_th = torch.where(
                (~fired_mask) & (self.v_th > self.v_threshold),
                torch.full_like(self.v_th, self.v_threshold),
                self.v_th
            )

    def extra_repr(self):
        return super().extra_repr() + f', alpha={self.alpha}'
