import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from soul.neuron import functional

__all__ = ['MetaSpikformer', 'MetaSpikformer2', 'MetaSpikformer4', 'MetaSpikformer8']

class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    
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
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguoys()

        return x

class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=False):
        super().__init__()

        conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

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
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
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
        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))
        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = deepcopy(lif)
        self.k_lif = deepcopy(lif)
        self.v_lif = deepcopy(lif)

        self.attn_lif = deepcopy(lif)
        self.attn_lif.v_threshold = 0.5

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3) # -> (T, B, C, N)
        q = q.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.k_lif(k).flatten(3)
        k = k.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        v = self.v_lif(k).flatten(3)
        v = v.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
        
        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x
    
class Block(nn.Module):
    def __init__(self, lif, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()

        self.attn = Attention(lif, dim, num_heads=num_heads)

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(lif, in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

class MetaSpikformer(nn.Module):
    def __init__(self, config, depths=[6, 2], embed_dims=[128, 256, 512, 640], norm_layer=nn.LayerNorm(eps=1e-6), drop_path_rate=0.0):
        super().__init__()

        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = config['input_channels']
        lif = config['neuron']

        mlp_ratio = config['mlp_ratio']
        num_heads = config['num_heads']

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

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
            embed_dims=embed_dims[2],
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
                mlp_ratio=mlp_ratio,
                drop_path=dpr[j],
                norm_layer=norm_layer,
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
            Block(lif, 
                  dim=embed_dims[3], 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio,
                  drop_path=dpr[j],
                  norm_layer=norm_layer,
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
        assert len(x.shape) in [4, 5], f'Invalid input shape {x.shape}...'
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]  

        x = self.forward_features(x) 
        x = self.forward_head(x)

        return x


def MetaSpikformer2(config): # 2-256
    return MetaSpikformer(config, depths=[1, 1], embed_dims=[64, 128, 256, 320])

def MetaSpikformer4(config): # 4-384
    return MetaSpikformer(config, depths=[3, 1], embed_dims=[96, 192, 384, 480])

def MetaSpikformer8(config): # 8-512
    return MetaSpikformer(config, depths=[6, 2], embed_dims=[128, 256, 512, 640]) 