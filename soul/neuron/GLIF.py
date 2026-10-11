"""
Filename:
    GLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-03

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Xingting Yao et al., "GLIF: A Uniﬁed Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", NeurIPS'2022.
    https://github.com/CAS-CLab/Gated-LIF
"""
import math

import numpy as np
import torch
import torch.nn as nn

from soul.neuron.base import MemoryModule, surrogate_from_config


class GatedLIFNode(MemoryModule):
    """
    Gated LIF 神经元：三个可学习门控 alpha / beta / gamma（取 sigmoid）分别在指数/线性泄漏、
    电导/直接输入、乘性/减性重置之间做插值；膜时间常数 tau、阈值 v_threshold、减性重置量 v_subreset、
    线性泄漏 linear_decay 以及逐时间步的电导 conduct 均可学习，以 logit 形式存储，使用时取 sigmoid。

    与 BaseNode 的逐步模板不同，门控在整段序列上计算，因此只支持多步模式，
    且输入的时间步数必须等于构造时的 time_step。

    配置项：time_step, init_tau, init_v_threshold, init_conduct（均在 (0, 1) 内）,
           init_linear_decay（缺省 init_v_threshold / (2 * time_step)）, init_v_subreset（缺省 init_v_threshold）
    """
    def __init__(self, config):
        super().__init__()
        T = config['time_step']
        init_tau = config['init_tau']
        init_v_threshold = config['init_v_threshold']
        init_conduct = config['init_conduct']
        init_linear_decay = config.get('init_linear_decay')
        init_v_subreset = config.get('init_v_subreset')

        assert isinstance(init_tau, float) and init_tau < 1.
        assert isinstance(T, int)
        assert (isinstance(init_linear_decay, float) and init_linear_decay < 1.) or init_linear_decay is None
        assert (isinstance(init_v_subreset, float) and init_v_subreset < 1.) or init_v_subreset is None

        self.surrogate_function = surrogate_from_config(config)
        self.T = T
        self.register_memory('v', 0.)
        self.register_memory('u', 0.)

        # layer-wise learnable params
        self.alpha, self.beta, self.gamma = [nn.Parameter(torch.tensor(0.2 * (np.random.rand() - 0.5), dtype=torch.float)) for i in range(3)]
        self.tau = nn.Parameter(torch.tensor(- math.log(1 / init_tau - 1), dtype=torch.float))
        self.v_threshold = nn.Parameter(torch.tensor(- math.log(1 / init_v_threshold - 1), dtype=torch.float))
        init_linear_decay = init_v_threshold / (T * 2) if init_linear_decay is None else init_linear_decay
        self.linear_decay = nn.Parameter(torch.tensor(- math.log(1 / init_linear_decay - 1), dtype=torch.float))
        init_v_subreset = init_v_threshold if init_v_subreset is None else init_v_subreset
        self.v_subreset = nn.Parameter(torch.tensor(- math.log(1 / init_v_subreset - 1), dtype=torch.float))
        self.conduct = nn.Parameter(- math.log(1 / init_conduct - 1) * torch.ones(T, dtype=torch.float))

    def supported_step_mode(self):
        return ('m',)

    def set_v_threshold(self, value: float):
        """
        把发放阈值 sigmoid(v_threshold) 设为 value，阈值仍可学习

        Args:
            value: 阈值大小，取值 (0, 1)

        Returns:
            None
        """
        assert 0. < value < 1., f'the threshold of {self._get_name()} must be in (0, 1), but got {value}'
        with torch.no_grad():
            self.v_threshold.fill_(- math.log(1 / value - 1))

    def extra_repr(self):
        with torch.no_grad():
            tau = self.tau.sigmoid().item()
            v_subreset = self.v_subreset.sigmoid().item()
            linear_decay = self.linear_decay.sigmoid().item()
            conduct = [round(c, 4) for c in self.conduct.sigmoid().tolist()]
        return super().extra_repr() + f', T={self.T}, tau={tau:.4f}, v_subreset={v_subreset:.4f}, linear_decay={linear_decay:.4f}, conduct={conduct}'

    def neuronal_charge(self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, t):
        input = x * (1 - beta * (1 - self.conduct[t].view(*self.target_shape).sigmoid()))
        self.u = ((1 - alpha * (1 - self.tau.view(*self.target_shape).sigmoid())) * self.v \
                  - (1 - alpha) * self.linear_decay.view(*self.target_shape).sigmoid()) \
                 + input

    def neuronal_reset(self, spike, alpha: torch.Tensor, gamma: torch.Tensor):
        self.u = self.u - (1 - alpha * (1 - self.tau.view(*self.target_shape).sigmoid())) * self.v * gamma * spike \
                 - (1 - gamma) * self.v_subreset.view(*self.target_shape).sigmoid() * spike

    def neuronal_fire(self):
        return self.surrogate_function(self.u - self.v_threshold.view(*self.target_shape).sigmoid())

    def multi_step_forward(self, x_seq: torch.Tensor):
        if x_seq.shape[0] != self.T:
            raise ValueError(f'{self._get_name()} is built for time_step={self.T}, but got an input with {x_seq.shape[0]} time steps')
        self.target_shape = (1, -1, *([1] * (x_seq.ndim - 3)))  # (1, -1, 1, 1) for conv2d or (1, -1) for linear

        alpha, beta, gamma = self.alpha.view(*self.target_shape).sigmoid(), self.beta.view(*self.target_shape).sigmoid(), self.gamma.view(*self.target_shape).sigmoid()
        y_seq = []
        spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
        for t in range(self.T):
            self.neuronal_charge(x_seq[t], alpha, beta, t)
            self.neuronal_reset(spike, alpha, gamma)
            spike = self.neuronal_fire()
            self.v = self.u
            y_seq.append(spike)
        return torch.stack(y_seq)
