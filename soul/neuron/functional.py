"""
Filename: functional.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-04-21
Description:
    implementation of resent function for the membran potential of spiking neurons after forward.

References:
    - Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023.
    https://github.com/fangwei123456/spikingjelly
"""
import torch.nn as nn
from soul.neuron import base
import logging

def reset_net(net: nn.Module):
    for m in net.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, base.MemoryModule):
                logging.warning(f'Trying to call `reset()` of {m}, which is not spikingjelly.activation_based.base'
                                f'.MemoryModule')
            m.reset()

def set_step_mode(net: nn.Module, step_mode: str):
    for m in net.modules():
        if not isinstance(m, (base.StepModule)):
            logging.warning(f'Trying to set the step mode for {m}, which is not spikingjelly.activation_based'
                            f'.base.StepModule')
        m.step_mode = step_mode
