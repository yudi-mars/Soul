"""
Filename: spikingresformer.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-06-20
Description:
    implementation for a transformer-structured SNN model for image classification.

References:
    - Xinyu Shi et al., "SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks", CVPR'2024.
      https://github.com/xyshi2000/SpikingResformer
"""
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List

from soul.neuron import functional

__all__ = ['SpikingResformer', 'SpikingResformer192', 'SpikingResformer256', 'SpikingResformer384', 'SpikingResformer512']

def multi_time_forward(x_seq, stateless_module):
    y_shape = [x_seq.shape[0], x_seq.shape[1]] # [T, B]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    
    y_shape.extend(y.shape[1:]) # [T, B] + [...] -> [T, B, ...]
    return y.view(y_shape)

class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class GWFFN(nn.Module):
    def __init__(self, lif, in_channels, num_conv=1, ratio=4, group_size=64):
        super().__init__()
        inner_channels = in_channels * ratio
        self.up_lif = deepcopy(lif)
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(inner_channels),
        )
        # self.conv = nn.ModuleList()
        self.num_conv = num_conv
        for n in range(num_conv):
            setattr(self, f'conv_lif{n}', deepcopy(lif))
            setattr(
                self, 
                f'conv{n}', 
                nn.Sequential(
                    nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, groups=inner_channels // group_size, bias=False),
                    nn.BatchNorm2d(inner_channels),
                )
            )

        self.down_lif = deepcopy(lif)
        self.down = nn.Sequential(
            nn.Conv2d(inner_channels, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()

        x = self.up_lif(x)

        x = multi_time_forward(x, self.up)

        x_feat_in = x.clone()
        for n in range(self.num_conv):
            x = getattr(self, f'conv_lif{n}')(x)
            x = multi_time_forward(x, getattr(self, f'conv{n}'))

        x = x + x_feat_in

        x = self.down_lif(x)
        x = multi_time_forward(x, self.down)
        x = x + x_feat_out

        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, lif, in_channels, out_channels, stride=2):
        super().__init__()
        self.sn = deepcopy(lif)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sn(x)
        x = multi_time_forward(x, self.conv)
        x = multi_time_forward(x, self.norm)

        return x
    
class DSSA(nn.Module):
    def __init__(self, lif, dim, num_heads, lenth, patch_size):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.lenth = lenth

        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))

        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = deepcopy(lif)

        self.W = nn.Conv2d(dim, 2 * dim, patch_size, patch_size, bias=False)
        self.norm = nn.BatchNorm2d(2 * dim)
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')

        self.activation_attn = deepcopy(lif)
        self.activation_out = deepcopy(lif)

        self.Wproj = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm_proj = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)

        y = multi_time_forward(x, self.W)
        y = multi_time_forward(y, self.norm)
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (1 - self.momentum)

        scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        attn = self.matmul1(y1.transpose(-1, -2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (1 - self.momentum)

        scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)
        out = self.matmul2(y2, attn)
        out = out * scale2
        out = out.reshape(T, B, C, H, W)
        out = self.activation_out(out)

        out = multi_time_forward(out, self.Wproj)
        out = multi_time_forward(out, self.norm_proj)
        out = out + x_feat

        return out

    
class SpikingResformer(nn.Module):
    def __init__(
        self, 
        config, 
        layers: List[List[str]],
        planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
    ):
        super().__init__()

        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = config['input_channels']
        img_size_h = config['input_height']
        img_size_w = config['input_width']
        assert img_size_w == img_size_h
        lif = config['neuron']

        group_size = config['group_size']
        mlp_ratio = config['mlp_ratio']

        self.prologue = nn.Sequential(
            nn.Conv2d(in_channels, planes[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(planes[0]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        img_size = img_size_h // 4

        # this is for cifar10, dvs data reproducible
        # self.prologue = nn.Sequential(
        #     nn.Conv2d(in_channels, planes[0], 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(planes[0]),
        # )
        # img_size = img_size_h

        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)

        self.layers = nn.Sequential()
        for idx in range(len(planes)):
            sub_layers = nn.Sequential()
            if idx != 0:
                sub_layers.append(
                    DownsampleLayer(lif, planes[idx - 1], planes[idx], stride=2)
                )
                img_size = img_size // 2
            for name in layers[idx]:
                if name == 'DSSA':
                    sub_layers.append(
                        DSSA(lif, planes[idx], num_heads[idx], (img_size // patch_sizes[idx]) ** 2, patch_sizes[idx])
                    )
                elif name == 'GWFFN':
                    sub_layers.append(
                        GWFFN(lif, planes[idx], group_size=group_size, ratio=mlp_ratio)
                    )
                else:
                    raise ValueError(name)
            self.layers.append(sub_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(planes[-1], num_classes, bias=False)
        
        self.init_weight()

    def forward_features(self, x):
        functional.reset_net(self)

        x = multi_time_forward(x, self.prologue)
        x = self.layers(x)
        x = multi_time_forward(x, self.avgpool)
        
        return x

    def forward_head(self, x):
        x = torch.flatten(x, 2)
        x = self.head(x).mean(0)

        return x

    def forward(self, x):
        x = self.forward_features(x) # [T, B, D]  
        x = self.forward_head(x)

        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def SpikingResformer192(config):
    return SpikingResformer(
        config,
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3,
        ],
        [64, 192, 384],
        [1, 3, 6],
        [4, 2, 1],
    )

def SpikingResformer256(config):
    return SpikingResformer(
        config,
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 256, 612],
        [1, 3, 6],
        [4, 2, 1]
    )

def SpikingResformer384(config):
    return SpikingResformer(
        config,
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 384, 768],
        [1, 6, 12],
        [4, 2, 1]
    )

def SpikingResformer512(config):
    return SpikingResformer(
        config,
        [
            ['DSSA', 'GWFFN'] * 1,
            ['DSSA', 'GWFFN'] * 2,
            ['DSSA', 'GWFFN'] * 3, ],
        [64, 512, 1024],
        [2, 8, 16],
        [4, 2, 1]
    )