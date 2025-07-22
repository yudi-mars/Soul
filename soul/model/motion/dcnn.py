"""
Filename: dcnn.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-15
Description:
    implementation for spike-wise DCNN-structured model for HAR.

References:
    - Yang Jianbo et al., "Deep Convolutional Neural Networks On Multichannel Time Series For Human Activity Recognition", IJCAI'2015.
      https://github.com/gongchenooo/MobiCom24-Delta
"""
import torch
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

class DCNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config['input_channels']
        input_dim = config['input_dim']
        lif = config['neuron']
        num_classes = config['num_classes']

        mlp_hidden_dim = config['mlp_hidden_dim']

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 50, kernel_size=5, bias=False),
            nn.BatchNorm1d(50)
        )
        self.lif1 = deepcopy(lif)
        input_dim -= 4

        self.subsamp1 = nn.MaxPool1d(kernel_size=2, bias=False)
        input_dim //= 2

        self.conv2 = nn.Sequential(
            nn.Conv1d(50, 40, kernel_size=5, bias=False),
            nn.BatchNorm1d(40)
        )
        self.lif2 = deepcopy(lif)
        input_dim -= 4

        self.subsamp2 = nn.MaxPool1d(kernel_size=2)
        input_dim //= 2

        if input_dim <= 20:
            self.conv3 = nn.Sequential(
                nn.Conv1d(40, 20, kernel_size=2, bias=False),
                nn.BatchNorm1d(20)
            )
            input_dim  -= 1
        else:
            self.conv3 = nn.Sequential(
                nn.Conv1d(40, 20, kernel_size=3, bias=False),
                nn.BatchNorm1d(20)
            )
            input_dim  -= 2
        self.lif3 = deepcopy(lif)

        self.linear = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim, bias=False),
            nn.BatchNorm1d(20)
        )
        self.lif4 = deepcopy(lif)

        self.head = nn.Linear(mlp_hidden_dim, num_classes)

    def forward_features(self, x):
        # x.shape: (T, B, C, D)
        functional.reset_net(self)

        x = multi_time_forward(x, self.conv1)
        x = self.lif1(x)

        x = multi_time_forward(x, self.subsamp1)

        x = multi_time_forward(x, self.conv2)
        x = self.lif2(x)

        x = multi_time_forward(x, self.subsamp2)

        x = multi_time_forward(x, self.conv3)
        x = self.lif3(x)

        return x # (T, B, C, D)
    
    def forward_head(self, x):
        x = multi_time_forward(x, self.linear) # (T, B, C, D) 
        x = torch.sum(x, dim=2) # -> (T, B, D)
        x = self.lif4(x)

        x = self.head(x.mean(0)) # -> (B, D)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x

        

