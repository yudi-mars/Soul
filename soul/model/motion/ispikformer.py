"""
Filename: is.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-23
Description:
    implementation for spike-wise iTransformer for HAR.

References:
    - Changze Lv. et al., "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks", ICML'2024.
      https://github.com/microsoft/SeqSNN
"""
import torch.nn as nn

from copy import deepcopy
from soul.neuron import functional

class DataEmbeddingInverted(nn.Module):
    def __init__(self, lif, c_in, d_model):
        super().__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.lif = deepcopy(lif)

    def forward(self, x):
        # x: T B L C
        T, B, _, C = x.shape
        x = x.permute(0, 1, 3, 2).flatten(0, 1)  # TB C L
        x = self.value_embedding(x)  # TB C H
        x = self.bn(x.transpose(-1, -2)).transpose(-1, -2)  # TB C H
        x = x.reshape(T, B, C, self.d_model)
        x = self.lif(x)  # T B C H
        return x
    
class Block(nn.Module):
    def __init__(self, lif, dim, d_ff, heads=8):
        super().__init__()
        self.attn = SSA(lif, dim=dim, heads=heads)
        self.mlp = MLP(lif, in_features=dim, hidden_features=d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
    
class SSA(nn.Module):
    def __init__(self, lif, dim, heads=8):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        self.qk_scale = 0.25

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = deepcopy(lif)

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = deepcopy(lif)

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = deepcopy(lif)

        self.attn_lif = deepcopy(lif)
        self.attn_lif.v_threshold = 0.5

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = deepcopy(lif)

    def forward(self, x):
        # x = x.transpose(0, 1)

        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1)  # TB L D
        q_m_out = self.q_m(x_for_qkv)  # TB L D
        q_m_out = (
            self.q_bn(q_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        q_m_out = self.q_lif(q_m_out)
        q = (
            q_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = (
            self.k_bn(k_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        k_m_out = self.k_lif(k_m_out)
        k = (
            k_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = (
            self.v_bn(v_m_out.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        v_m_out = self.v_lif(v_m_out)
        v = (
            v_m_out.reshape(T, B, L, self.heads, D // self.heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale
        x = attn @ v  # x_shape: T * B * heads * L * D//heads

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)

        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())

        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        lif,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()

        out_features = out_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = deepcopy(lif)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = deepcopy(lif)

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.flatten(0, 1)  # TB L D
        x = self.fc1(x)  # TB L H
        x = (
            self.bn1(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, self.hidden_features)
            .contiguous()
        )
        x = self.lif1(x)
        x = x.flatten(0, 1)  # TB L H
        x = self.fc2(x)  # TB L D
        x = (
            self.bn2(x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(T, B, L, D)
            .contiguous()
        )
        x = self.lif2(x)
        return x
    
class ISpikformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_channels = config['input_channels']
        input_dim = config['input_dim']
        lif = config['neuron']
        num_classes = config['num_classes']

        embedding_dim = config['embedding_dim']
        num_heads = config['num_head']
        depths = config['depth']

        d_ff = int(embedding_dim * config['mlp_ratio'])

        self.emb = DataEmbeddingInverted(lif, input_dim, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                Block(
                    lif, 
                    dim=embedding_dim,
                    d_ff=d_ff,
                    heads=num_heads,
                )
                for _ in range(depths)
            ]
        )

        self.head = nn.Linear(input_channels * embedding_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x
    
    def forward_features(self, x):
        functional.reset_net(self)

        x = x.transpose(2, 3) # (T, B, C, L) -> (T, B, L, C)
        x = self.emb(x)
        for blk in self.blocks:
            x = blk(x)  # (T, B, C, H)

        return x
    
    def forward_head(self, x):
        x = x.mean(0)
        x = self.head(x.flatten(1))

        return x
