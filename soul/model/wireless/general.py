"""
Filename:
    general.py

Author:
    Di Yu <yudi2023@zju.edu.cn>

Date Created:
    2025-08-09

Description:
    implementation for commonly-used network architectures in edge scenarios.
    These models are customized for wireless sensing tasks.

References:
    - Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.
    https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
"""
import torch.nn as nn
from soul.utils import multi_time_forward
from copy import deepcopy

from soul.neuron import functional

__all__ = ['SpikingMLP', 'SpikingLeNet', 'SpikingRNN', 'SpikingConvRNN']


class SpikeRNNCell(nn.Module):
    '''
    SNN is a variant of RNN
    '''
    def __init__(self, lif, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.lif = deepcopy(lif)

    def forward(self, x):
        x =  multi_time_forward(x, self.linear) # (T, B, L, D) -> (T, B, L, D')
        x = self.lif(x)  
        return x

class SpikingMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        C, H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.ln1 = nn.Linear(C * H * W, config['hidden_dim'])
        self.lif1 = deepcopy(lif)

        self.ln2 = nn.Linear(config['hidden_dim'], config['hidden_dim'])
        self.lif2 = deepcopy(lif)

        self.head = nn.Linear(config['hidden_dim'], self.num_classes)

    def forward(self, x):
        functional.reset_net(self)

        x = x.flatten(2)  # (T, B, C, H, W) -> (T, B, CHW)

        x = multi_time_forward(x, self.ln1)
        x = self.lif1(x)

        x = multi_time_forward(x, self.ln2)
        x = self.lif2(x)

        x = self.head(x.mean(0)) # (T, B, D) -> (B, D)

        return x

class SpikingLeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        C, H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.conv1 = nn.Conv2d(C, 32, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = deepcopy(lif)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        H = (H - 4) // 2
        W = (W - 4) // 2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = deepcopy(lif)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        H = (H - 4) // 2
        W = (W - 4) // 2

        self.conv3 = nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(96)
        self.lif3 = deepcopy(lif)

        H -= 4
        W -= 4

        self.ln1 = nn.Linear(96 * H * W, config['hidden_dim'])
        self.lif4 = deepcopy(lif)

        self.head = nn.Linear(config['hidden_dim'], self.num_classes)

    def forward(self, x):
        functional.reset_net(self)

        x = multi_time_forward(x, [self.conv1, self.bn1])
        x = self.lif1(x)

        x = multi_time_forward(x, self.pool1)

        x = multi_time_forward(x, [self.conv2, self.bn2])
        x = self.lif2(x)

        x = multi_time_forward(x, self.pool2)

        x = multi_time_forward(x, [self.conv3, self.bn3])
        x = self.lif3(x)

        x = x.flatten(2) # (T, B, C, H, W) -> (T, B, CHW)

        x = multi_time_forward(x, self.ln1)
        x = self.lif4(x)

        x = self.head(x.mean(0)) # (T, B, D) -> (B, num_cls)

        return x
        
class SpikingRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        # H will be the sequence length
        C, self.H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.input_size = C * W
        output_size = config['hidden_dim']
        num_layers = config['num_layers']
        self.use_last_step = config['last_step']

        self.rnn = []
        for l in range(num_layers):
            if l == 0:
                self.rnn.append(SpikeRNNCell(lif, self.input_size, output_size))
            else:
                self.rnn.append(SpikeRNNCell(lif, output_size, output_size))
        self.rnn = nn.Sequential(*self.rnn)

        self.head = nn.Linear(output_size, self.num_classes)

    def forward(self, x):
        functional.reset_net(self)

        x = x.permute(0, 1, 3, 4, 2) # (T, B, C, H, W) -> (T, B, H, W, C)
        x = x.reshape(x.size(0), x.size(1), self.H, self.input_size) # (T, B, H, W, C) -> (T, B, H, WC)

        x = self.rnn(x) # -> (T, B, L, D),  L=H
        
        if self.use_last_step: # -> (T, B, D)
            x = x[:, :, -1, :]
        else:
            x = x.mean(dim=2)

        out = self.head(x.mean(0)) # -> (B, D) -> (B, num_cls)

        return out
    
class SpikingConvRNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        C, H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.conv1 = nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = deepcopy(lif)

        self.pool1 = nn.MaxPool2d((2, 2))
        H //= 2
        W //= 2

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = deepcopy(lif)

        self.pool2 = nn.MaxPool2d((2, 2))
        H //= 2
        W //= 2

        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.lif3 = deepcopy(lif)

        output_size = config['hidden_dim']
        num_layers = config['num_layers']
        self.use_last_step = config['last_step']

        self.rnn = []
        for l in range(num_layers):
            if l == 0:
                self.rnn.append(SpikeRNNCell(lif, 96 * W, output_size))
            else:
                self.rnn.append(SpikeRNNCell(lif, output_size, output_size))
        self.rnn = nn.Sequential(*self.rnn)

        self.head = nn.Linear(output_size, self.num_classes)

    def forward(self, x):
        functional.reset_net(self)

        x = multi_time_forward(x, [self.conv1, self.bn1])
        x = self.lif1(x)

        x = multi_time_forward(x, self.pool1)

        x = multi_time_forward(x, [self.conv2, self.bn2])
        x = self.lif2(x)

        x = multi_time_forward(x, self.pool2)

        x = multi_time_forward(x, [self.conv3, self.bn3])
        x = self.lif3(x) # -> (T, B, C, H, W)

        x = x.permute(0, 1, 3, 4, 2).contiguous() # -> (T, B, H, W, C)
        x = x.flatten(-2) # -> (T, B, H, WC)

        x = self.rnn(x) # -> (T, B, L, D),  L=H
        if self.use_last_step: # -> (T, B, D)
            x = x[:, :, -1, :]
        else:
            x = x.mean(dim=2)

        out = self.head(x.mean(0)) # -> (B, D) -> (B, num_cls)

        return out