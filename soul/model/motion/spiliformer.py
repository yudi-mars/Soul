"""
Filename:
    spiliformer.py

Author:
    Di Yu <yudi2023@zju.edu.cn>

Date Created:
    2026-03-05

Description:
    implementation for a transformer-structured SNN model for image classification.

References:
    - Zeqi Zheng et al., "SpiLiFormer: Enhancing Spiking Transformers with Lateral Inhibition", ICCV'2025.
      https://github.com/KirinZheng/SpiLiFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from soul.neuron import functional

__all__ = ['Spike_Lateral_Transformer', 'SpiLiFormer256', 'SpiLiFormer384', 'SpiLiFormer512']

class Decoder(nn.Module):
    def __init__(self, lif, in_channels, out_channels, T):
        super().__init__()

        self.decoder_bn = nn.BatchNorm1d(out_channels)

        self.out_channels = out_channels
        self.decoder_linear = nn.Linear(in_channels, out_channels,bias=False)
 
        self.T = T
        
        self.decoder_lif = deepcopy(lif)


    def forward(self,x):   
        _ ,N,C = x.shape

        x = self.decoder_linear(x).transpose(-1, -2).contiguous()

        x = self.decoder_bn(x).reshape(self.T, -1, C, N).transpose(-1, -2).contiguous()

        x = self.decoder_lif(x).reshape(-1,C,N).transpose(-1,-2).contiguous()
        return x
    
class FF_LiDiff_Attention(nn.Module):
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
        self.attn_lif.v_threshold = 0.5

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = deepcopy(lif)


    def forward(self, x):

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, 2 * self.num_heads, C // self.num_heads // 2, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = q.reshape(T, B, 2, self.num_heads, C // self.num_heads // 2, N).contiguous()
        q_1, q_2 = torch.sum(q[:, :, 0], dim=3, keepdim=True), torch.sum(q[:, :, 1], dim=3, keepdim=True)
        attn = q_1 - q_2
        attn = self.attn_lif(attn)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, N)
        x = self.proj_lif(x)

        return x
    
class FB_LiDiff_Attention(nn.Module):
    def __init__(self, lif, dim, num_heads=8):
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
        self.attn_lif.v_threshold = 0.5

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = deepcopy(lif)

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x, fb=None):
        T, B, C, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T,B,C,N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T,B,C,N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        if fb is not None:
            fb = self.v_conv(fb.transpose(-2, -1))
            v_conv_out = v_conv_out + fb
        v_conv_out = self.v_bn(v_conv_out).reshape(T,B,C,N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = k.transpose(-2,-1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0,1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T,B,C,N)

        return x

class MLP(nn.Module):
    def __init__(self, lif, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.mlp1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1, bias=False)
        self.mlp1_bn = nn.BatchNorm1d(hidden_features)
        self.mlp1_lif = deepcopy(lif)

        self.mlp2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1, bias=False)
        self.mlp2_bn = nn.BatchNorm1d(out_features)
        self.mlp2_lif = deepcopy(lif)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        # x: (T, B, C, N)
        T, B, C, N = x.shape

        x = x.flatten(0, 1).contiguous()
        x = self.mlp1_conv(x)
        x = self.mlp1_bn(x)
        x = x.reshape(T, B, self.c_hidden, N).contiguous()
        x = self.mlp1_lif(x)

        x = x.flatten(0, 1).contiguous()
        x = self.mlp2_conv(x)
        x = self.mlp2_bn(x)
        x = x.reshape(T, B, self.c_output, N).contiguous()
        x = self.mlp2_lif(x)

        return x

class FF_LiDiff_Transformer(nn.Module):
    def __init__(self, lif, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.tssa = FF_LiDiff_Attention(lif, dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(lif, in_features= dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x

class FB_LiDiff_Transformer(nn.Module):
    def __init__(self, lif, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.ssa = FB_LiDiff_Attention(lif, dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(lif, in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x, feedback=None):

        x = x + self.ssa(x, feedback)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, lif, input_dim=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.W = input_dim // patch_size[0]        
        self.num_patches = self.W
        self.proj_conv = nn.Conv1d(in_channels, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims // 2)
        self.proj_lif = deepcopy(lif)

        self.proj1_conv = nn.Conv1d(embed_dims // 2, embed_dims // 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm1d(embed_dims // 1)
        self.proj1_lif = deepcopy(lif)

        self.proj_res_conv = nn.Conv1d(embed_dims//2, embed_dims //1, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm1d(embed_dims)
        self.proj_res_lif = deepcopy(lif)

    def forward(self, x):
        T, B, C, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        w0 = x.shape[-1]
        x = self.proj_bn(x).reshape(T, B, -1, w0).contiguous()
        x = self.proj_lif(x).flatten(0, 1)

        x_feat = x
        x = self.proj1_conv(x)
        w1 = x.shape[-1]
        x = self.proj1_bn(x).reshape(T, B, -1, w1).contiguous()
        x = self.proj1_lif(x)

        n = x.shape[-1]
        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, n).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat # shortcut

        return x
    
class PatchEmbeddingStage(nn.Module):
    def __init__(self, lif, input_dim=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.W = input_dim // patch_size[0]
        self.num_patches = self.W

        self.proj3_conv = nn.Conv1d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm1d(embed_dims)
        self.proj3_lif = deepcopy(lif)

        self.proj4_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm1d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = deepcopy(lif)

        self.proj_res_conv = nn.Conv1d(embed_dims//2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm1d(embed_dims)
        self.proj_res_lif = deepcopy(lif)

    def forward(self, x):
        # x: (T, B, C, N)
        T, B, C, N = x.shape

        x = x.flatten(0, 1).contiguous()   # (T*B, C, N)
        x_feat = x

        # main branch
        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        n1 = x.shape[-1]
        x = x.reshape(T, B, -1, n1).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x)
        n2 = x.shape[-1]
        x = x.reshape(T, B, -1, n2).contiguous()
        x = self.proj4_lif(x)

        # shortcut branch
        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        n_res = x_feat.shape[-1]
        x_feat = x_feat.reshape(T, B, -1, n_res).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat
        return x

class Spike_Lateral_Transformer(nn.Module):
    def __init__(self, config, depths=[6, 8, 6], embed_dims=[64, 128, 256]):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']
        self.input_channels = int(config["input_channels"])
        self.input_dim = int(config["input_dim"])

        mlp_ratios = config['mlp_ratio']
        patch_size = config['patch_size']
        lif = config['neuron']

        self.depths = depths
        num_heads = [config['num_heads'], config['num_heads'], config['num_heads']]

        patch_embed1 = PatchEmbedInit(
            lif,
            input_dim=self.input_dim,
            patch_size=patch_size,
            in_channels=self.input_channels,
            embed_dims=embed_dims // 4
        )

        stage1 = nn.ModuleList([
            FF_LiDiff_Transformer(
                lif,
                dim=embed_dims // 4, 
                num_heads=num_heads[0], 
                mlp_ratio=mlp_ratios
            ) for _ in range(1)
        ])

        patch_embed2 = PatchEmbeddingStage(
            lif, 
            input_dim=self.input_dim,
            patch_size=patch_size,
            in_channels=self.input_channels,
            embed_dims=embed_dims // 2
        )

        stage2 = nn.ModuleList([
            FF_LiDiff_Transformer(
                lif,
                dim=embed_dims // 2, 
                num_heads=num_heads[1], 
                mlp_ratio=mlp_ratios
            ) for _ in range(1)
        ])

        patch_embed3 = PatchEmbeddingStage(
            lif, 
            input_dim=self.input_dim,
            patch_size=patch_size,
            in_channels=self.input_channels,
            embed_dims=embed_dims
        )

        stage3 = nn.ModuleList([
            FB_LiDiff_Transformer(
                lif, 
                dim=embed_dims, 
                num_heads=num_heads[2], 
                mlp_ratio=mlp_ratios
            ) for _ in range(depths - 2)
        ])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"patch_embed3", patch_embed3)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)
        setattr(self, f"stage3", stage3)

        # classification head
        self.head = nn.Linear(embed_dims, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.prompt = nn.Parameter(torch.randn(embed_dims), requires_grad=True)
        self.prompt_2 = nn.Parameter(torch.randn(embed_dims), requires_grad=True)
        self.decoder_stage_3 = nn.ModuleList([
            Decoder(lif, embed_dims, embed_dims, self.T) for _ in range(depths - 2)
        ])
        self.aug = 1.0

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def feedback(self,x):
        # T, B, N, C = x.shape
    
        fb = []

        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        cos_sim_2 = F.normalize(x, dim=-1) @ F.normalize(self.prompt_2[None, ..., None], dim=1)

        mask = cos_sim.clamp(0,1)
        mask_2 = cos_sim_2.clamp(0,1)

        x = (x * mask - x * mask_2)* self.aug
        x = x.flatten(0,1)
        for i in range(len(self.decoder_stage_3) - 1, -1, -1):
            out = self.decoder_stage_3[i](x)
            fb = [out] + fb
            x = out

        return fb

    def forward_features(self, x, second_forward = None):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")
        stage3 = getattr(self, f"stage3")
        patch_embed3 = getattr(self, f"patch_embed3")
        tmp = None
        if second_forward is None:
            x = patch_embed1(x)  # T, B, N, C
            for blk in stage1:
                x = blk(x)
            
            x = patch_embed2(x)
            for blk in stage2:
                x = blk(x)
            
            x = patch_embed3(x)
            tmp = x

        second_forward_len = len(second_forward) if second_forward is not None else 0
        start_index = self.depths - 2 - second_forward_len
        for i, blk in enumerate(stage3):
            if second_forward is not None:
                if i < start_index:
                    feedback_i = None
                else:
                    feedback_i = second_forward[i - start_index]
            else:
                feedback_i = None
            x = blk(x, feedback_i) if feedback_i is not None else blk(x)
        return x, tmp
    
    def first_forward(self, x):
        functional.reset_net(self)
        # x -> (T, B, N, C)
        x, tmp = self.forward_features(x)
        second_forward_information = self.feedback(x.flatten(3).transpose(-2, -1)) # -> (T, B, N, C)
        return self.head(x.flatten(3).mean(3).mean(0)), second_forward_information, tmp
    
    def second_forward(self, x, second_forward):
        functional.reset_net(self)

        x, _ = self.forward_features(x, second_forward)
        x = self.head(x.flatten(3).mean(3).mean(0))

        return x

    def forward(self, x):
        # first forward
        x1, feedback, tmp = self.first_forward(x)
        # second forward
        x = self.second_forward(tmp, feedback)
        
        if self.training:
            return x, x1
        else:
            return x
    
def SpiLiFormer256(config):
    return Spike_Lateral_Transformer(config, depths=4, embed_dims=256)

def SpiLiFormer384(config):
    return Spike_Lateral_Transformer(config, depths=6, embed_dims=384)

def SpiLiFormer512(config):
    return Spike_Lateral_Transformer(config, depths=8, embed_dims=512)
