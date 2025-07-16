"""
Filename: dcnn.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-16
Description:
    implementation for spike-wise SenseHAR.

References:
    - Jeyakumar J.V. et al., "SenseHAR: A Robust Virtual Activity Sensor for Smartphones and Wearables", 2019.
      https://github.com/devanshuDesai/SenseHAR
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
    return y.reshape(y_shape)

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
    necessarily precedes the waist sensor. Therefore, when aggregating embeddings, using a linear layer with mean operation 
    is more appropriate than an LSTM.
    '''
    def __init__(self, lif, in_channels=3, emb_dim=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
        )
        self.lif1 = deepcopy(lif)

        self.mp = nn.MaxPool1d(2)

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
        )
        self.lif2 = deepcopy(lif)

        self.ap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        # x.shape (T, BD, in_C, L)
        x = multi_time_forward(x, self.conv1) 
        x = self.lif1(x)

        x = multi_time_forward(x, self.mp)

        x = multi_time_forward(x, self.conv2)
        x = self.lif2(x) # -> (T, BD, C, out_D)

        x = multi_time_forward(x, self.ap).squeeze(-1) # -> (T, BD, C, 1) -> (T, BD, C)

        x = multi_time_forward(x, self.fc) # -> (T, BD, dim)

        return x

class SenseHAR(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_channels = config['input_channels']
        input_dim = config['input_dim']
        lif = config['neuron']
        num_classes = config['num_classes']

        emb_dim = config['embedding_dim']
        mlp_hidden_dim = config['mlp_hidden_dim']

        # input_channels = num_devices * 1 or num_devices * 3
        if input_channels % 3 == 0: 
            self.num_devices = int(input_channels / 3)
        else:
            self.num_devices = input_channels

        self.input_channels = 1 if input_channels % 3 else 3

        # All devices share the same encoder, enabling the model to learn shared abstractions across devices.
        self.encoder = DeviceEncoder(lif, self.input_channels, emb_dim)

        self.app = nn.Linear(emb_dim, mlp_hidden_dim)
        self.app_lif = deepcopy(lif)

        self.head = nn.Linear(mlp_hidden_dim, num_classes)

    def forward_features(self, x):
        functional.reset_net(self)

        T, B, C, L = x.shape
        assert C == self.num_devices * self.input_channels

        x = x.reshape(T, B * self.num_devices, self.input_channels, L)

        emb = self.encoder(x) # -> (T, BD, dim)
        emb = emb.reshape(T, B, self.num_devices, -1) # -> (T, B, D, dim)

        # aggregation all embedding
        agg = emb.mean(dim=2) # -> (T, B, dim)

        return agg
    
    def forward_head(self, x):
        x = multi_time_forward(x, self.app)
        x = self.app_lif(x) # -> (T, B, D)
        x = x.mean(0) # -> (B, D)
        
        x = self.head(x)

        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x