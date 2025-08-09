"""
Filename: general.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-08-09
Description:
    implementation for commonly-used network architectures in edge scenarios.
    These models are customized for vision sensing tasks.

References:
    - Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.
    https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
"""
import torch.nn as nn

from copy import deepcopy

from soul.neuron import functional

__all__ = ['SpikingMLP', 'SpikingLeNet', 'SpikingRNN', 'SpikingGRU', 'SpikingLSTM', 'SpikingConvLSTM']

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

class SpikingMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        C, H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.ln1 = nn.Linear(C * H * W, 1024)
        self.lif1 = deepcopy(lif)

        self.ln2 = nn.Linear(1024, 1024)
        self.lif2 = deepcopy(lif)

        self.head = nn.Linear(1024, self.num_classes)

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

        self.ln1 = nn.Linear(96 * H * W, 512)
        self.lif4 = deepcopy(lif)

        self.head = nn.Linear(512, self.num_classes)

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

        x = self.flatten(2) # (T, B, C, H, W) -> (T, B, CHW)

        x = multi_time_forward(x, self.ln1)
        x = self.lif4(x)

        x = self.head(x.mean(0)) # (T, B, D) -> (B, num_cls)

        return x
        
class SpikingLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        hidden_size = config['hidden_size']

        # H will be the sequence length
        C, self.H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.input_size = C * W

        self.lstm = nn.LSTM(self.input_size, hidden_size, batch_first=True)

        self.head = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x):
        functional.reset_net(self)

        x = x.permute(0, 1, 3, 4, 2) # (T, B, C, H, W) -> (T, B, H, W, C)
        x = x.reshape(x.size(0), x.size(1), self.H, self.input_size) # (T, B, H, W, C) -> (T, B, H, WC)

        # TODO
        out, (hn, cn) = self.lstm(x)
        last = hn[-1] 

        outputs = self.head(last)

        return outputs
    
class SpikingConvLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        hidden_size = config['hidden_size']

        C, H, W = config['input_channels'], config['input_height'], config['input_width']
        lif = config['neuron']

        self.conv1 = nn.Conv2d(C, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool1d

    def forward(self, x):
        functional.reset_net(self)



        return x

