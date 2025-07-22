"""
Filename: dcl.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-22
Description:
    spike-wise deep convolutional and LSTM network for HAR, (hybrid model with ANN and SNN)

References:
    - Ordóñez, F. J. et al., "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition", Sensors'2016.
    - Yuhang Li et al., "Wearable-based Human Activity Recognition with Spatio-Temporal Spiking Neural Networks", Frontiers in Neuroscience'2023
    https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR
"""
import torch.nn as nn

from copy import deepcopy

from soul.neuron import functional

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

class DCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        in_channels = config['input_channels']
        self.input_dim = config['input_dim']
        lif = config['neuron']
        num_classes = config['num_classes']

        hidden_dim = config['hidden_dim'] 
        n_channels = config['n_channels']

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=(1, 5), padding='same', bias=False),
            nn.BatchNorm2d(n_channels),
        )
        self.lif1 = deepcopy(lif)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=(1, 5), padding='same', bias=False),
            nn.BatchNorm2d(n_channels),
        )
        self.lif2 = deepcopy(lif)

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=(1, 5), padding='same', bias=False),
            nn.BatchNorm2d(n_channels),
        )
        self.lif3 = deepcopy(lif)

        self.conv4 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=(1, 5), padding='same', bias=False),
            nn.BatchNorm2d(n_channels),
        )
        self.lif4 = deepcopy(lif)

        self.lstm = nn.LSTM(n_channels * in_channels, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, x):
        functional.reset_net(self)
        x = x.unsqueeze(2) # -> (T, B, 1, L, D)
        
        x = multi_time_forward(x, self.conv1)
        x = self.lif1(x)
        x = multi_time_forward(x, self.conv2)
        x = self.lif2(x)
        x = multi_time_forward(x, self.conv3)
        x = self.lif3(x)
        x = multi_time_forward(x, self.conv4)
        x = self.lif4(x)

        return x 
    
    def forward_head(self, x):
        x = x.mean(0).permute(0, 3, 1, 2).flatten(2) # -> (B, D, C * L)

        x, _ = self.lstm(x)
        x = self.head(x[:, -1, :]) # use the last time step output

        return x  # (B, num_classes)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x
