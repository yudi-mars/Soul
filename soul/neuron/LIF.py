"""
Filename:
    LIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-04-29

Description:
    implementation for LIF for Spiking Neural Networks.

References:
    - Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023.
    https://github.com/fangwei123456/spikingjelly
"""
import torch

from soul.neuron.base import BaseNode


class LIFNode(BaseNode):
    """
    Leaky Integrate-and-Fire 神经元，以 v_reset 为静息电位（软重置时为 0）：
        decay_input=True:  v <- v + (v_reset - v + x) / tau
        decay_input=False: v <- v + (v_reset - v) / tau + x

    额外配置项：tau（膜时间常数）, decay_input（缺省 True）
    """
    def __init__(self, config):
        super().__init__(config)
        self.tau = config['tau']
        self.decay_input = config.get('decay_input', True)

    def neuronal_charge(self, x: torch.Tensor):
        v_rest = 0. if self.v_reset is None else self.v_reset
        if self.decay_input:
            self.v = self.v + (v_rest - self.v + x) / self.tau
        else:
            self.v = self.v + (v_rest - self.v) / self.tau + x

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, decay_input={self.decay_input}'
