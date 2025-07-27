"""
Filename: LTMD.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-05-09
Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Siqi Wang et al., "LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout", NeurIPS'2022.
    https://github.com/sq117/LTMD
"""
import torch
import torch.nn as nn

class SpikeAct(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.gt(input, 0)
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        hu = abs(input) < 0.5
        hu = hu.float() / (2 * 0.5)
        return grad_input * hu

spikeAct = SpikeAct.apply

def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n1, thre):
    u_t1_n1 = 0.25 * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n1
    v_t1_n1 = u_t1_n1 - thre
    o_t1_n1 = spikeAct(v_t1_n1)
    return u_t1_n1, o_t1_n1

class LTMD(nn.Module):
    def __init__(self, cofig):
        super(LTMD, self).__init__()
        init_w = cofig['kappa']
        self.steps = cofig['steps']
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
        
    def forward(self, x):
        x = x.permute(1, 2, 3, 4, 0)
        u = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(self.steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step], self.w.tanh())
        out = out.permute(4, 0, 1, 2, 3)
        return out
