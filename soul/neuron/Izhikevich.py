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
from soul.neuron.LIF import BaseNode
from typing import Callable, Optional
import torch
from soul.utils.surrogate import Sigmoid

class AdaptBaseNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: Optional[float] = 0.,
                 v_rest: float = 0., w_rest: float = 0., tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        # b: jump amplitudes
        # a: subthreshold coupling
        assert isinstance(w_rest, float)
        assert isinstance(v_rest, float)
        assert isinstance(tau_w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.register_memory('w', w_rest)

        self.w_rest = w_rest
        self.v_rest = v_rest
        self.tau_w = tau_w
        self.a = a
        self.b = b

    @staticmethod
    @torch.jit.script
    def jit_neuronal_adaptation(w: torch.Tensor, tau_w: float, a: float, v_rest: float, v: torch.Tensor):
        return w + 1. / tau_w * (a * (v - v_rest) - w)

    def neuronal_adaptation(self):
        self.w = self.jit_neuronal_adaptation(self.w, self.tau_w, self.a, self.v_rest, self.v)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, w: torch.Tensor, spike_d: torch.Tensor, v_reset: float, b: float,
                       spike: torch.Tensor):
        v = (1. - spike_d) * v + spike * v_reset
        w = w + b * spike
        return v, w

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, w: torch.Tensor, spike_d: torch.Tensor, v_threshold: float, b: float,
                       spike: torch.Tensor):
        v = v - spike_d * v_threshold
        w = w + b * spike
        return v, w

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v, self.w = self.jit_soft_reset(self.v, self.w, spike_d, self.v_threshold, self.b, spike)
        else:
            # hard reset
            self.v, self.w = self.jit_hard_reset(self.v, self.w, spike_d, self.v_reset, self.b, spike)

    def extra_repr(self):
        return super().extra_repr() + f', v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.w_float_to_tensor(x)
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            w_init = self.w
            self.w = torch.full_like(x.data, fill_value=w_init)

class IzhikevichNode(AdaptBaseNode):
    def __init__(self, config):
                #  tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1.,
                #  v_reset: Optional[float] = 0., v_rest: float = -0.1, w_rest: float = 0., tau_w: float = 2., a: float = 0.,
                #  b: float = 0.,
                #  surrogate_function: Callable = Sigmoid(), detach_reset: bool = False, step_mode='m', store_v_seq: bool = False):
        self.tau = config["tau"]
        self.v_c = config["v_c"]
        self.a0 = config["a0"]
        self.v_threshold = config["v_threshold"]
        self.v_reset = config["v_reset"]
        self.v_rest = config["v_rest"]
        self.w_rest = config["w_rest"]
        self.tau_w = config["tau_w"]
        self.a = config["a"]
        self.b = config["b"]
        self.surrogate_function = config['surrogate']
        self.detach_reset = config['detach_reset']
        self.step_mode = config['step_mode']
        self.store_v_seq = config['store_v_seq']

        assert isinstance(self.tau, float) and self.tau > 1.
        assert self.a0 > 0
    
        super().__init__(self.v_threshold, self.v_reset, self.v_rest, self.w_rest, self.tau_w, self.a, self.b, self.surrogate_function, self.detach_reset, self.step_mode, self.store_v_seq)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c) - self.w) / self.tau

    def multi_step_forward(self, x_seq: torch.Tensor):
        super().multi_step_forward(x_seq)
