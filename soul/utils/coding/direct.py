"""
Filename:
    direct.py

Author:
    Di Yu <yudi2023@zju.edu.cn>

Date Created:
    2025-04-09

Description:
    implementation of direct coding mechanism for SNN inputs.
"""
import torch

def encode(data, num_steps):
    """
    direct coding for inputs by repeating roughly

    Args:
        data: torch.Tensor
            Data tensor for a single batch of shape [batch x input_size]
        num_steps:

    Returns:
        torch.Tensor
                rate encoding spike train of input features of shape [num_steps x batch x input_size]
    """
    data = data.unsqueeze(0).repeat(tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist()))

    return data # (T, B, C, H, W)
