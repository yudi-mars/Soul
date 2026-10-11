"""
Filename:
    ILIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    TBD

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Zhenxin Lei et al., "ILIF: Temporal Inhibitory Leaky Integrate-and-Fire Neuron for Overactivation in Spiking Neural Networks", IJCAI'2025.
    https://github.com/kaisun1/ILIF
"""
import torch

from soul.neuron.LIF import LIFNode


class ILIFNeuron(LIFNode):
    """
    Inhibitory LIF 神经元：
        抑制电流 inhibition <- 0.03 * (inhibition + x_{t-1} * spike_{t-1})，充电输入为 x - relu(inhibition)；
        LIF 重置后累积抑制记忆 h <- h + spike * v，再执行 v <- v - spike * sigmoid(h)。

    配置项同 LIFNode
    """
    def __init__(self, config):
        super().__init__(config)
        self.register_memory('inhibitory_memory', 0.)
        self.register_memory('prev_input', 0.)
        self.register_memory('prev_spike', 0.)
        self.register_memory('inhibition', 0.)

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        if isinstance(self.prev_input, float):
            self.prev_input = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)
            self.inhibition = torch.zeros_like(x)

        self.inhibition = 0.03 * (self.inhibition + self.prev_input * self.prev_spike)
        self.neuronal_charge(x - torch.clamp(self.inhibition, min=0))
        self.prev_input = x
        spike = self.neuronal_fire()
        self.prev_spike = spike
        self.neuronal_reset(spike)
        self.inhibitory_memory = self.inhibitory_memory + spike * self.v
        self.v = self.v - spike * torch.sigmoid(self.inhibitory_memory)
        return spike
