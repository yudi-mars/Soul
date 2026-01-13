"""
Filename: metaspikeformer.py
Author: Weisong Zhang <22551070@zju.edu.cn>
Date Created: 2026-01-02
Description:
    Adaptation for a transformer-structured SNN model for audio classification.

    Note that the original RepConv will cause the misconvergence of all models. 
    Besides, the two convolution layers are linked without any neurons, which is not
    a legal computation logic in neuromorphic chips. Hence, we use a normal linear layer
    to replace the RepConv part for a stable implementation.

References:
    - Man Yao et al., "Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips", ICLR'2024.
      https://github.com/BICLab/Spike-Driven-Transformer-V2
"""
import torch.nn as nn
from copy import deepcopy

from soul.neuron import functional

__all__ = ['MetaSpikeformer', 'MetaSpikeformer256', 'MetaSpikeformer384', 'MetaSpikeformer512']
    
class DownSampling(nn.Module):
    def __init__(self, lif, in_channels, embed_dim, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()

        self.encode_conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.encode_bn = nn.BatchNorm2d(embed_dim)
        if not first_layer:
            self.encode_lif = deepcopy(lif)

    def forward(self, x):
        T, B, _, _, _ = x.shape
        if hasattr(self, 'encode_lif'):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        lif, 
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = deepcopy(lif)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = deepcopy(lif)
        self.dwconv = nn.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = x.flatten(0, 1)
        x = self.bn1(self.pwconv1(x)).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = x.flatten(0, 1)
        x = self.dwconv(x)
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)

        return x
    
class ConvBlock(nn.Module):
    def __init__(self, lif, dim, mlp_ratio=4):
        super().__init__()

        self.Conv = SepConv(lif, dim)
        self.lif1 = deepcopy(lif)
        self.conv1 = nn.Conv2d(
            dim, dim * mlp_ratio, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.lif2 = deepcopy(lif)
        self.conv2 = nn.Conv2d(
            dim * mlp_ratio, dim, kernel_size=3, padding=1, groups=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(dim)

        self.mlp_ratio = mlp_ratio

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, self.mlp_ratio * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x
    
class MLP(nn.Module):
    def __init__(self, lif, in_features, hidden_features=None, out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = deepcopy(lif)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = deepcopy(lif)

        self.c_hidden = hidden_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x
    
class Attention(nn.Module):
    def __init__(self, lif, dim, num_heads=8):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = deepcopy(lif)

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)


        self.q_lif = deepcopy(lif)
        self.k_lif = deepcopy(lif)
        self.v_lif = deepcopy(lif)

        self.attn_lif = deepcopy(lif)
        self.attn_lif.v_threshold = 0.5

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = self.head_lif(x)

        T, B, C, H, W = x.shape
        N = H * W

        x = x.reshape(T, B, C, N).transpose(-1, -2)

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C)

        x = x.transpose(2, 3).reshape(T, B, C, H, W).contiguous()

        return x
    
class Block(nn.Module):
    def __init__(self, lif, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        self.attn = Attention(lif, dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(lif, in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class MetaSpikeformer(nn.Module):
    def __init__(self, config, depths=[6, 2], embed_dims=[128, 256, 512, 640]):
        super().__init__()

        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = 1
        lif = config['neuron']

        mlp_ratio = config['mlp_ratio']
        num_heads = config['num_heads']

        self.downsample1_1 = DownSampling(
            lif, 
            in_channels=in_channels,
            embed_dim=embed_dims[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.convblock1_1 = nn.ModuleList(
            [ConvBlock(lif, dim=embed_dims[0] // 2, mlp_ratio=mlp_ratio)]
        )

        self.downsample1_2 = DownSampling(
            lif,
            in_channels=embed_dims[0] // 2,
            embed_dim=embed_dims[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.convblock1_2 = nn.ModuleList(
            [ConvBlock(lif, dim=embed_dims[0], mlp_ratio=mlp_ratio)]
        )

        self.downsample2 = DownSampling(
            lif,
            in_channels=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.convblock2_1 = nn.ModuleList(
            [ConvBlock(lif, dim=embed_dims[1], mlp_ratio=mlp_ratio)]
        )

        self.convblock2_2 = nn.ModuleList(
            [ConvBlock(lif, dim=embed_dims[1], mlp_ratio=mlp_ratio)]
        )

        self.downsample3 = DownSampling(
            lif,
            in_channels=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList([
            Block(
                lif, 
                dim=embed_dims[2], 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio
            ) for j in range(depths[0])
        ])

        self.downsample4 = DownSampling(
            lif,
            in_channels=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )

        self.block4 = nn.ModuleList([
            Block(
                lif, 
                dim=embed_dims[3], 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio
            ) for j in range(depths[1])
        ])

        self.lif = deepcopy(lif)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = x.unsqueeze(2)
        functional.reset_net(self)

        x = self.downsample1_1(x)
        for blk in self.convblock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.convblock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.convblock2_1:
            x = blk(x)
        for blk in self.convblock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)

        return x # -> (T, B, C, N)


    def forward_head(self, x):
        x = x.flatten(3).mean(3) # ->(T, B, C)
        x = self.lif(x).mean(0) # -> (B, C)

        x = self.head(x) # -> (B, num_cls)

        return x

    def forward(self, x):

        x = self.forward_features(x) 
        x = self.forward_head(x)

        return x


def MetaSpikeformer256(config): # 2-256
    return MetaSpikeformer(config, depths=[1, 1], embed_dims=[64, 128, 256, 320])

def MetaSpikeformer384(config): # 4-384
    return MetaSpikeformer(config, depths=[2, 2], embed_dims=[96, 192, 384, 480])

def MetaSpikeformer512(config): # 8-512
    return MetaSpikeformer(config, depths=[6, 2], embed_dims=[128, 256, 512, 640]) 