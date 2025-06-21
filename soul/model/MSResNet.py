import torch
import torch.nn as nn
from copy import deepcopy
from soul.neuron import functional

__all__ = ['MSResNet', 'MSResNet18', 'MSResNet34', 'MSResNet50']

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

class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, lif, in_channels, out_channels, stride=1, base_width=64, groups=1):
        super().__init__()

        width = int(out_channels * (base_width / 64.)) * groups

        self.sn1 = deepcopy(lif)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(width),
        )
        self.sn2 = deepcopy(lif)
        self.conv2 = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, stride=stride, groups=groups, bias=False),
            nn.BatchNorm2d(width)
        )
        self.sn3 = deepcopy(lif)
        self.conv3 = nn.Sequential(
            nn.Conv2d(width, out_channels * BottleBlock.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * BottleBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BottleBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleBlock.expansion)
            )

    def forward(self, x):
        identity =  multi_time_forward(x, self.shortcut)

        x = self.sn1(x)
        x = multi_time_forward(x, self.conv1)
        x = self.sn2(x)
        x = multi_time_forward(x, self.conv2)
        x = self.sn3(x)
        x = multi_time_forward(x, self.conv3)

        return x + identity


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, lif, in_channels, out_channels, stride=1):
        super().__init__()

        self.sn1 = deepcopy(lif),
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),

        )
        self.sn2 = deepcopy(lif)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        identity = multi_time_forward(x, self.shortcut)

        x = self.sn1(x)
        x = multi_time_forward(x, self.conv1)
        x = self.sn2(x)
        x = multi_time_forward(x, self.conv2)

        return x + identity

class MSResNet(nn.Module):
    def __init__(self, config, block, layers):
        super().__init__()

        lif = config['neuron']
        num_classes = config['num_classes']
        self.T = config['time_step']
        in_channels = config['input_channels']

        k = 1
        self.in_planes = 64 * k

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                       self.in_planes,
                       kernel_size=7,
                       padding=3,
                       bias=False,
                       stride=2),
            nn.BatchNorm2d(self.in_planes),
        )
        self.sn1 = deepcopy(lif)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = self._make_layer(block, 64 * k, layers[0], 2)
        self.conv3_x = self._make_layer(block, 128 * k, layers[1], 2)
        self.conv4_x = self._make_layer(block, 256 * k, layers[2], 2)
        self.conv5_x = self._make_layer(block, 512 * k, layers[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512 * block.expansion * k, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = multi_time_forward(x, self.conv1)
        x = multi_time_forward(x, self.maxpool)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.sn1(x)

        return x

    def forward_head(self, x):
        x = multi_time_forward(x, self.avgpool)
        x = torch.flatten(x, 2).mean(0)
        x = self.head(x)

        return x
    
    def forward(self, x):
        functional.reset_net(self)

        assert len(x.shape) in [4, 5], f'Invalid input shape {x.shape}...'
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1) # B, T, C, H, W
        x = x.transpose(0, 1) # [T, B, C, H, W]

        x = self.forward_features(x)
        x = self.forward_head(x)

        return x
    
def MSResNet18(config):
    return MSResNet(config, BasicBlock, [2, 2, 2, 2])

def MSResNet34(config):
    return MSResNet(config, BasicBlock, [3, 4, 6, 3])

def MSResNet50(config):
    return MSResNet(config, BottleBlock, [3, 4, 6, 3])
