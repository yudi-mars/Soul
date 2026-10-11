"""
Filename:
    CLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-01

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Yulong Huang et al., "CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", ICML'2024.
    https://github.com/HuuYuLong/Complementary-LIF
"""
import torch

from soul.neuron.LIF import LIFNode


class CLIFNode(LIFNode):
    """
    Complementary LIF 神经元：在 LIF 的基础上维护互补记忆 m，
        充电后 m <- m * sigmoid(v / tau)，发放后 m <- m + spike，
        LIF 重置后再执行 v <- v - spike * sigmoid(m)。

    配置项同 LIFNode
    """
    def __init__(self, config):
        super().__init__(config)
        self.register_memory('m', 0.)

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        self.m = self.m * torch.sigmoid(self.v / self.tau)
        spike = self.neuronal_fire()
        self.m = self.m + spike
        self.neuronal_reset(spike)
        self.v = self.v - spike * torch.sigmoid(self.m)
        return spike
