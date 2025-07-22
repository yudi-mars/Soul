"""
Filename: sensehar.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-16
Description:
    implementation for spike-wise SenseHAR for HAR.

References:
    - Jeyakumar J.V. et al., "SenseHAR: A Robust Virtual Activity Sensor for Smartphones and Wearables", SenSys'2019.
      https://github.com/devanshuDesai/SenseHAR
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

class DeviceEncoder(nn.Module):
    '''
    Sensor Fusion Model for device input 
    stage 1 contains 1-D Conv layer to extract features from each sensor stream
    stage 2 captures the correlation across the corresponding axes in different sensors
    stage 3 fuses the extracted features from the previous stages to a low-dimensional latent space

    In this repository, the selected dataset are collected by different motion sensors (e.g., accelerator, gyroscope). 
    Usually, these sensors collected inputs with 3-dimensions, i.e., x, y, and z. Therefore, except UCI HAR, which used 
    processed data as input, the other three dataset contains features whose dimension can be divided evenly by 3.

    Meanwhile, there is no temporal dependency between devices in this reposotory. For example, the left-wrist band never 
    necessarily precedes the waist sensor. Therefore, when aggregating embeddings, using two linear layer with mean operation 
    is more appropriate than an 2-layer LSTM.
    '''
    def __init__(self, lif, in_channels, K=2, encoder_channels=64, share_channels=3):
        super().__init__()

        self.num_sensors = 1 if in_channels % share_channels else int(in_channels / share_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, encoder_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(encoder_channels),
        )
        self.lif1 = deepcopy(lif)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_channels * self.num_sensors, encoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(encoder_channels),
        )
        self.lif2 = deepcopy(lif)

        self.mp = nn.MaxPool2d((1, K))

        self.time_dist = nn.Linear(encoder_channels * self.num_sensors, self.num_sensors, bias=False)
        self.time_lif = deepcopy(lif)

    def forward(self, x):
        # stage 1
        xs = torch.split(x, self.num_sensors, dim=2) # (T, B, C, L) -> num * (T, B, C / num, L)

        temp = []
        for x in xs:
            x = x.unsqueeze(2) # -> (T, B, 1, C / num, L)
            x = multi_time_forward(x, self.conv1) # (T, B, 1, C / num, L) -> (T, B, C_1, C / num, L)
            temp.append(x)

        # stage 2
        x = torch.concatenate(temp, dim=2) # -> (T, B, num * C_1, C / num, L) 
        x = self.lif1(x)

        x = multi_time_forward(x, self.conv2) # -> (T, B, C_2, C / num, L)
        x = self.lif2(x)

        # stage 3
        x = multi_time_forward(x, self.mp) # -> (T, B, C_2, C / num, L // K)
        T, B, C, N, D = x.shape
        x = x.reshape(T, B, C * N, D).contiguous() # -> (T, B, F, D), F = C_2 * C / num, D = L // K

        T, B, F, D = x.shape
        x_ = x.permute(0, 1, 3, 2).reshape(T, -1, F) # -> (T, B, D, F) -> (T, B * D, F)
        x = multi_time_forward(x_, self.time_dist) # -> (T, B * D, F')
        x = self.time_lif(x)

        x = x.reshape(T, B, D, -1).permute(0, 1, 3, 2) # -> (T, B, D, F') -> (T, B, F', D)

        return x

class SenseHAR(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_channels = config['input_channels']
        input_dim = config['input_dim']
        lif = config['neuron']
        num_classes = config['num_classes']

        K = config['k_pool']
        mlp_hidden_dim = config['mlp_hidden_dim']
        encoder_channels = config['encoder_channels']

        # input_channels = num_devices * 1 or num_devices * 3
        if input_channels % 3 == 0: 
            self.num_devices = int(input_channels / 3)
        else:
            self.num_devices = input_channels

        self.input_channels = 1 if input_channels % 3 else 3

        self.encoder = DeviceEncoder(lif, input_channels, K, encoder_channels)

        self.app_ln1 = nn.Linear(input_dim // K, mlp_hidden_dim, bias=False)
        self.app_lif1 = deepcopy(lif)
        self.app_ln2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim, bias=False)
        self.app_lif2 = deepcopy(lif)

        self.head = nn.Linear(mlp_hidden_dim, num_classes)

    def forward_features(self, x):
        functional.reset_net(self)

        emb = self.encoder(x) # -> (T, B, L, D)

        # aggregation all embedding
        agg = emb.mean(dim=2) # -> (T, B, D)
        
        return agg
    
    def forward_head(self, x):
        x = multi_time_forward(x, self.app_ln1)
        x = self.app_lif1(x) # -> (T, B, D)
        x = multi_time_forward(x, self.app_ln2)
        x = self.app_lif2(x) 
        x = x.mean(0) # -> (B, D)
        
        x = self.head(x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x