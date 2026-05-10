"""
Filename:
    spikingformer.py

Author:
    Weisong Zhang <22551070@zju.edu.cn>

Date Created:
    2026-05-10

Description:
    Implementation of Spikingformer (spike-driven Transformer SNN) adapted for Soul.

    Key differences from Spikformer:
    - Attention uses Conv1d (spike-driven) instead of Linear for Q/K/V
    - MLP uses Conv2d 1x1 on spatial feature maps (T,B,C,H,W)
    - 5-block hierarchical tokenizer with MaxPool downsampling
    - Feature maps stay in (T,B,C,H,W) format throughout

References:
    - Zhaokun Zhou et al., "Spikingformer: when spiking neural network meets transformer", ICLR'2023.
      https://github.com/ZK-Zhou/spikformer
"""
import torch
import torch.nn as nn
from copy import deepcopy

from soul.neuron import functional

__all__ = ['Spikingformer', 'Spikingformer256', 'Spikingformer384', 'Spikingformer512']


class DropPath(nn.Module):
    """Stochastic depth (drop path) per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class SpikingformerMLP(nn.Module):
    """MLP block using Conv2d 1x1 on spatial feature maps (T, B, C, H, W)."""
    def __init__(self, lif, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = deepcopy(lif)
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = deepcopy(lif)
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x


class SpikingformerAttention(nn.Module):
    """Spike-driven self-attention using Conv1d for Q/K/V projections."""
    def __init__(self, lif, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = deepcopy(lif)
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = deepcopy(lif)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = deepcopy(lif)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = deepcopy(lif)

        self.attn_lif = deepcopy(lif)
        self.attn_lif.v_threshold = 0.5

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x)

        x = x.flatten(3)  # (T, B, C, H*W)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)  # (T*B, C, N)

        q = self.q_conv(x_for_qkv)
        q = self.q_bn(q).reshape(T, B, C, N)
        q = self.q_lif(q)
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        k = self.k_conv(x_for_qkv)
        k = self.k_bn(k).reshape(T, B, C, N)
        k = self.k_lif(k)
        k = k.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        v = self.v_conv(x_for_qkv)
        v = self.v_bn(v).reshape(T, B, C, N)
        v = self.v_lif(v)
        v = v.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125

        x = x.transpose(3, 4).reshape(T, B, C, N)
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x


class SpikingformerBlock(nn.Module):
    """Transformer block with residual connections."""
    def __init__(self, lif, dim, num_heads, mlp_ratio=4., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpikingformerAttention(lif, dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SpikingformerMLP(lif, in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikingformerTokenizer(nn.Module):
    """5-block hierarchical CNN tokenizer with MaxPool downsampling."""
    def __init__(self, lif, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        self.patch_size = (patch_size, patch_size)
        self.C = in_channels
        self.H = img_size_h // self.patch_size[0]
        self.W = img_size_w // self.patch_size[1]
        self.num_patches = self.H * self.W

        self.block0_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.block0_bn = nn.BatchNorm2d(embed_dims // 8)

        self.block1_lif = deepcopy(lif)
        self.block1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.block1_bn = nn.BatchNorm2d(embed_dims // 4)

        self.block2_lif = deepcopy(lif)
        self.block2_conv = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2_bn = nn.BatchNorm2d(embed_dims // 2)

        self.block3_lif = deepcopy(lif)
        self.block3_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block3_bn = nn.BatchNorm2d(embed_dims)

        self.block4_lif = deepcopy(lif)
        self.block4_mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.block4_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.block0_conv(x.flatten(0, 1))
        x = self.block0_bn(x).reshape(T, B, -1, H, W)

        x = self.block1_lif(x).flatten(0, 1)
        x = self.block1_conv(x)
        x = self.block1_bn(x).reshape(T, B, -1, H, W)

        x = self.block2_lif(x).flatten(0, 1)
        x = self.block2_conv(x)
        x = self.block2_bn(x).reshape(T, B, -1, H, W)

        x = self.block3_lif(x).flatten(0, 1)
        x = self.block3_mp(x)
        x = self.block3_conv(x)
        H3, W3 = x.shape[2], x.shape[3]
        x = self.block3_bn(x).reshape(T, B, -1, H3, W3)

        x = self.block4_lif(x).flatten(0, 1)
        x = self.block4_mp(x)
        x = self.block4_conv(x)
        H4, W4 = x.shape[2], x.shape[3]
        x = self.block4_bn(x).reshape(T, B, -1, H4, W4)

        return x, (H4, W4)


class Spikingformer(nn.Module):
    """
    Spikingformer: Spike-driven Transformer SNN for image classification.

    Architecture: SpikingformerTokenizer -> SpikingformerBlock x N -> Linear head.
    All activations are LIF neurons (pure event-driven). Attention uses Conv1d
    and MLP uses Conv2d 1x1, maintaining (T, B, C, H, W) feature maps.
    """
    def __init__(self, config, depths=4, embed_dims=384, num_heads=None, drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = config['input_channels']
        img_size_h = config['input_height']
        img_size_w = config['input_width']
        lif = config['neuron']
        patch_size = config['patch_size']
        mlp_ratio = config['mlp_ratio']
        num_heads = num_heads or config['num_heads']

        self.num_classes = num_classes
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.patch_embed = SpikingformerTokenizer(
            lif,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )

        self.blocks = nn.ModuleList([
            SpikingformerBlock(
                lif, dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depths)
        ])

        self.head = nn.Linear(embed_dims, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        functional.reset_net(self)
        x, (H, W) = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return x.flatten(3).mean(3)  # (T, B, C) spatial average pooling

    def forward_head(self, x):
        return self.head(x.mean(0))  # temporal mean -> (B, num_classes)

    def forward(self, x):
        # Vision input: (T, B, C, H, W) from run_soul.py after transpose
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def Spikingformer256(config):
    return Spikingformer(config, depths=2, embed_dims=256, num_heads=8)

def Spikingformer384(config):
    return Spikingformer(config, depths=4, embed_dims=384, num_heads=12)

def Spikingformer512(config):
    return Spikingformer(config, depths=8, embed_dims=512, num_heads=8)
