"""
Filename: qkformer.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-06-25
Description:
    implementation for a transformer-structured SNN model for image classification.

References:
    - Chenlin Zhou et al., "QKFormer: Hierarchical Spiking Transformer using Q-K Attention", NeurIPS'2024.
      https://github.com/zhouchenlin2096/QKFormer
"""
import torch
import torch.nn as nn
from copy import deepcopy
from functools import partial

from soul.neuron import functional

__all__ = ['QKFormer', 'QKFormer256', 'QKFormer384', 'QKFormer512']

def set_v_threshold(neuron, value: float):
    vt = getattr(neuron, "v_threshold", None)
    if isinstance(vt, torch.nn.Parameter) or torch.is_tensor(vt):
        with torch.no_grad():
            vt.fill_(float(value))
    else:
        neuron.v_threshold = float(value)


class PatchEmbedInit(nn.Module):
    def __init__(self, lif, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()

        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 2)
        self.proj_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif = deepcopy(lif)

        self.proj1_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims)
        self.proj1_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = deepcopy(lif)

        self.proj2_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims)
        self.proj2_lif = deepcopy(lif)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = deepcopy(lif)

    # def forward(self, x):
    #     T, B, C, H, W = x.shape

    #     x = self.proj_conv(x.flatten(0, 1))
    #     x = self.proj_bn(x)
    #     x = self.proj_maxpool(x).reshape(T, B, -1, H//2, W//2).contiguous()
    #     x = self.proj_lif(x).flatten(0, 1).contiguous()

    #     x_feat = x
    #     x = self.proj1_conv(x)
    #     x = self.proj1_bn(x)
    #     x = self.proj1_maxpool(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
    #     x = self.proj1_lif(x).flatten(0, 1).contiguous()

    #     x = self.proj2_conv(x)
    #     x = self.proj2_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
    #     x = self.proj2_lif(x)

    #     x_feat = self.proj_res_conv(x_feat)
    #     x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//4, W//4).contiguous()
    #     x_feat = self.proj_res_lif(x_feat)

    #     x = x + x_feat

    #     return x
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)

        x = self.proj_maxpool(x)
        h2, w2 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, h2, w2).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x_feat = x  # TB, C, h2, w2

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)

        x = self.proj1_maxpool(x)
        h4, w4 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, h4, w4).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T, B, -1, h4, w4).contiguous()
        x = self.proj2_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        hf, wf = x_feat.shape[-2], x_feat.shape[-1]
        x_feat = x_feat.reshape(T, B, -1, hf, wf).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x

    
class Token_QK_Attention(nn.Module):
    def __init__(self, lif, dim, num_heads=8):
        super().__init__() 
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = deepcopy(lif)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = deepcopy(lif)

        self.attn_lif = deepcopy(lif)
        # self.attn_lif.v_threshold = 0.5
        self.attn_lif = deepcopy(lif)
        set_v_threshold(self.attn_lif, 0.5)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = deepcopy(lif)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)
        
        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x
    
class MLP(nn.Module):
    def __init__(self, lif, in_features, hidden_features=None, out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = deepcopy(lif)

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = deepcopy(lif)

        self.c_hidden = hidden_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        x = self.fc2_lif(x)

        return x
    
class TokenSpikingTransformer(nn.Module):
    def __init__(self, lif, dim, num_heads, mlp_ratio):
        super().__init__()

        self.attn = Token_QK_Attention(lif, dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(lif, in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x
    
class PatchEmbeddingStage(nn.Module):
    def __init__(self, lif, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()

        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.proj3_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = deepcopy(lif)

        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_lif = deepcopy(lif)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = deepcopy(lif)

    # def forward(self, x):
    #     T, B, C, H, W = x.shape

    #     x = x.flatten(0, 1).contiguous()
    #     x_feat = x

    #     x = self.proj3_conv(x)
    #     x = self.proj3_bn(x)
    #     x = self.proj3_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
    #     x = self.proj3_lif(x).flatten(0, 1).contiguous()

    #     x = self.proj4_conv(x)
    #     x = self.proj4_bn(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
    #     x = self.proj4_lif(x)

    #     x_feat = self.proj_res_conv(x_feat)
    #     x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H // 2, W // 2).contiguous()
    #     x_feat = self.proj_res_lif(x_feat)

    #     x = x + x_feat

    #     return x
    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(0, 1).contiguous()
        x_feat = x  # TB, C, H, W

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)

        x = self.proj3_maxpool(x)
        h2, w2 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, h2, w2).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x).reshape(T, B, -1, h2, w2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        hf, wf = x_feat.shape[-2], x_feat.shape[-1]
        x_feat = x_feat.reshape(T, B, -1, hf, wf).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x

    
class SSA(nn.Module):
    def __init__(self, lif, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = deepcopy(lif)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = deepcopy(lif)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = deepcopy(lif)

        self.attn_lif = deepcopy(lif)
        # self.attn_lif.v_threshold = 0.5
        self.attn_lif = deepcopy(lif)
        set_v_threshold(self.attn_lif, 0.5)


        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = deepcopy(lif)

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = x.flatten(3)
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        # x = x.flatten(0, 1)
        # x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, H, W)

        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x
    
class SpikingTransformer(nn.Module):
    def __init__(self, lif, dim, num_heads, mlp_ratio=4.):
        super().__init__()

        self.attn = SSA(lif, dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(lif, in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class QKFormer(nn.Module):
    def __init__(self, config, depths=4, embed_dims=384, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = config['input_channels']
        img_size_h = config['input_height']
        img_size_w = config['input_width']
        lif = config['neuron']

        num_heads = config['num_heads']
        mlp_ratio = config['mlp_ratio']
        patch_size = config['patch_size']

        assert depths >= 3
        self.depths = depths

        self.patch_embed1 = PatchEmbedInit(
            lif,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 4
        )

        self.stage1 = nn.ModuleList([
            TokenSpikingTransformer(
                lif,
                dim=embed_dims // 4, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio
            ) for _ in range(1)
        ])

        self.patch_embed2 = PatchEmbeddingStage(
            lif, 
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 2
        )

        self.stage2 = nn.ModuleList([
            TokenSpikingTransformer(
                lif,
                dim=embed_dims // 2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio
            ) for _ in range(2)
        ])

        self.patch_embed3 = PatchEmbeddingStage(
            lif, 
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims
        )

        self.stage3 = nn.ModuleList([
            SpikingTransformer(
                lif,
                dim=embed_dims, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio
            ) for _ in range(depths - 3)
        ])


        self.head = nn.Linear(embed_dims, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        functional.reset_net(self)

        x = self.patch_embed1(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.stage3:
            x = blk(x)

        return x.flatten(3).mean(3) # (T, B, D)

    def forward_head(self, x):
        return self.head(x.mean(0))

    def forward(self, x):
        x = self.forward_features(x) # [T, B, D]  
        x = self.forward_head(x)

        return x
    
def QKFormer256(config):
    return QKFormer(config, depths=4, embed_dims=256)

def QKFormer384(config):
    return QKFormer(config, depths=6, embed_dims=384)

def QKFormer512(config):
    return QKFormer(config, depths=8, embed_dims=512)
