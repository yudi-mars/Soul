"""
Filename:
    lenet_variants.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2026-10-12

Description:
    SpikingLeNet variants that implement the synaptic side of two neurons, mixed into the SpikingLeNet of
    every application:
        - DendriticLeNet: dendritic bilinear integration of DLIF;
        - DualPathLeNet:  separate current and conductance pathways of DGN.

References:
    - Qianyi Bai et al., "A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks", ICLR'2026.
"""
import copy

from soul.neuron import DGNNode, DendriticBilinear, functional
from soul.utils import multi_time_forward


def _with_channel_dim(x):
    # acoustic and motion inputs (T, B, H, W) have no channel dimension
    return x.unsqueeze(2) if x.dim() == 4 else x


def _fresh_copy(layer):
    layer = copy.deepcopy(layer)
    layer.reset_parameters()
    return layer


class DendriticLeNet:
    """
    DLIF 的 SpikingLeNet：前两层卷积的输出加上 DendriticBilinear 对同一输入的树突整合（按卷积输出的位置裁剪）；
    单通道输入的第一层没有通道间交互，不加树突整合。可以配合任意神经元使用，配合 dlif 即为 DLIF。
    """
    def __init__(self, config):
        super().__init__(config)
        self.bilinear1 = DendriticBilinear(self.conv1.in_channels, self.conv1.out_channels) if self.conv1.in_channels > 1 else None
        self.bilinear2 = DendriticBilinear(self.conv2.in_channels, self.conv2.out_channels)

    @staticmethod
    def dendritic_conv(x, conv, bilinear):
        y = multi_time_forward(x, conv)
        if bilinear is None:
            return y
        z = multi_time_forward(x, bilinear)
        # the convolutions have no padding, so their outputs sit (kernel_size - 1) // 2 inside the input
        crop = tuple(slice((k - 1) // 2, (k - 1) // 2 + n) for k, n in zip(conv.kernel_size, y.shape[3:]))
        return y + z[(Ellipsis, *crop)]

    def forward(self, x):
        functional.reset_net(self)
        x = _with_channel_dim(x)

        x = self.dendritic_conv(x, self.conv1, self.bilinear1)
        x = self.lif1(multi_time_forward(x, self.bn1))
        x = multi_time_forward(x, self.pool1)

        x = self.dendritic_conv(x, self.conv2, self.bilinear2)
        x = self.lif2(multi_time_forward(x, self.bn2))
        x = multi_time_forward(x, self.pool2)

        x = self.lif3(multi_time_forward(x, [self.conv3, self.bn3]))

        x = x.flatten(2) # (T, B, C, H, W) -> (T, B, CHW)
        x = self.lif4(multi_time_forward(x, self.ln1))

        return self.head(x.mean(0)) # (T, B, D) -> (B, num_cls)


class DualPathLeNet:
    """
    完整 DGN 的 SpikingLeNet：每个突触层旁边增加一条形状相同、独立初始化、不做 BN 的电导支路，
    神经元以 lif(电流, 电导) 的形式接收两路输入，因此只能配合 dgn 使用。
    """
    def __init__(self, config):
        if not isinstance(config['neuron'], DGNNode):
            raise ValueError(f'{type(self).__name__} feeds a conductance pathway to its neurons and needs the dgn neuron, '
                             f'but got {type(config["neuron"]).__name__}')
        super().__init__(config)
        self.conv1_g, self.conv2_g, self.conv3_g, self.ln1_g = (
            _fresh_copy(layer) for layer in (self.conv1, self.conv2, self.conv3, self.ln1))
        for m in self.modules():
            if isinstance(m, DGNNode):
                # the shared equilibrium potential is unused with a conductance input; DDP rejects unused trainable parameters
                m.equilibrium.requires_grad_(False)

    def forward(self, x):
        functional.reset_net(self)
        x = _with_channel_dim(x)

        x = self.lif1(multi_time_forward(x, [self.conv1, self.bn1]), multi_time_forward(x, self.conv1_g))
        x = multi_time_forward(x, self.pool1)

        x = self.lif2(multi_time_forward(x, [self.conv2, self.bn2]), multi_time_forward(x, self.conv2_g))
        x = multi_time_forward(x, self.pool2)

        x = self.lif3(multi_time_forward(x, [self.conv3, self.bn3]), multi_time_forward(x, self.conv3_g))

        x = x.flatten(2) # (T, B, C, H, W) -> (T, B, CHW)
        x = self.lif4(multi_time_forward(x, self.ln1), multi_time_forward(x, self.ln1_g))

        return self.head(x.mean(0)) # (T, B, D) -> (B, num_cls)
