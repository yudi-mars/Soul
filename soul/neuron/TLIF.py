"""
Filename:
    TLIF.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-05-17

Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Yufei Guo et al., "Ternary spike: Learning ternary spikes for spiking neural networks", AAAI'2024.
    https://github.com/yfguo91/Ternary-Spike
"""
from collections import ChainMap

from soul.neuron.INTLIF import INTLIFNode


class TLIFNode(INTLIFNode):
    """
    三值脉冲 LIF 神经元：脉冲 surrogate_function(v)（如 Ternary: round(clamp(v, -1, 1))）取值 {-1, 0, 1}，
    膜电位不泄漏（tau 固定为 1），输出 spike / 8，其余行为同 INTLIFNode。

    配置项：v_threshold, v_reset, surrogate, detach_reset
    """
    def __init__(self, config):
        super().__init__(ChainMap({'tau': 1., 'norm': 8}, config))
