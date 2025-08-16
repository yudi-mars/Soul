"""
Filename: ILIF.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: TBD
Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Zhenxin Lei et al., "ILIF: Temporal Inhibitory Leaky Integrate-and-Fire Neuron for Overactivation in Spiking Neural Networks", IJCAI'2025.
    https://github.com/kaisun1/ILIF
"""

import torch
from .LIF import LIFNode

class InhibitoryLIFNeuron(LIFNode):
    def __init__(self, config):
        super().__init__(config)
        
        self.tau = config['tau']
        self.v_threshold = config['v_threshold']
        self.v_reset = config['v_reset']
        self.surrogate_function = config['surrogate_function']
        self.detach_reset = config['detach_reset']
        self.decay_input = config['decay_input']
        
        self.register_memory('inhibitory_memory', 0.)  # Inhibitory memory
        self.register_memory('prev_input', 0.)
        self.register_memory('prev_spike', 0.)
        self.register_memory('inhibition', 0.)

    def forward(self, x: torch.Tensor):
        if isinstance(self.prev_input, float):
            self.prev_input = torch.zeros_like(x)
            self.prev_spike = torch.zeros_like(x)
            self.inhibition = torch.zeros_like(x)
        current_input = x
        self.inhibition = 0.03*(self.inhibition + self.prev_input * self.prev_spike)
        self.neuronal_charge(x-torch.clamp(self.inhibition,min=0))
        self.prev_input = current_input
        spike = self.neuronal_fire()
        self.prev_spike = spike
        self.neuronal_reset(spike)
        self.inhibitory_memory = 1*(self.inhibitory_memory + spike * self.v)
        self.v = self.v - spike * torch.sigmoid(self.inhibitory_memory)  # Reset
        return spike

    def neuronal_charge(self, x: torch.Tensor):
        self._charging_v(x)

    def neuronal_reset(self, spike: torch.Tensor):
        self._reset(spike)

    def _charging_v(self, x: torch.Tensor):
        if self.decay_input:
            x = x / self.tau
            
        if self.v_reset is None or self.v_reset == 0:
            if type(self.v) is float:
                self.v = x
            else:
                self.v = self.v * (1 - 1. / self.tau) + x
        else:
            if type(self.v) is float:
                self.v = self.v_reset * (1 - 1. / self.tau) + self.v_reset / self.tau + x
            else:
                self.v = self.v * (1 - 1. / self.tau) + self.v_reset / self.tau + x

    def _reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1. - spike) * self.v + spike * self.v_reset

#Multi-step version of InhibitoryLIFNeuron
class ILIFNeuron(InhibitoryLIFNeuron):
    def __init__(self, config):
        super().__init__(config)
        
        self.tau = config['tau']
        self.v_threshold = config['v_threshold']
        self.v_reset = config['v_reset']
        self.surrogate_function = config['surrogate_function']
        self.detach_reset = config['detach_reset']
        self.decay_input = config['decay_input']
        
    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() > 1
        # x_seq.shape = [T, *]
        spike_seq = []
        self.v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(super().forward(x_seq[t]).unsqueeze(0))
            self.v_seq.append(self.v.unsqueeze(0))
        spike_seq = torch.cat(spike_seq, 0)
        self.v_seq = torch.cat(self.v_seq, 0)
        return spike_seq