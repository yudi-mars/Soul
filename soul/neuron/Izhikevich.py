"""
Filename:
    Izhikevich.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-04-29

Description:
    implementation for Izhikevich Neuron for Spiking Neural Networks.

References:
    - E. M. Izhikevich, "Simple model of spiking neurons," in IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003, doi: 10.1109/TNN.2003.820440.
    - Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023.
    https://github.com/fangwei123456/spikingjelly
"""
import torch

from soul.neuron.base import AdaptBaseNode


class IzhikevichNode(AdaptBaseNode):
    """
    Izhikevich 神经元：v <- v + (x + a0 * (v - v_rest) * (v - v_c) - w) / tau，适应电流 w 的动力学见 AdaptBaseNode。

    额外配置项：tau（> 1）, v_c, a0（> 0），以及 AdaptBaseNode 的 v_rest, w_rest, tau_w, a, b
    """
    def __init__(self, config):
        super().__init__(config)
        self.tau = config['tau']
        self.v_c = config['v_c']
        self.a0 = config['a0']
        assert isinstance(self.tau, float) and self.tau > 1.
        assert self.a0 > 0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c) - self.w) / self.tau
