import torch
import torch.nn as nn

from copy import deepcopy

from soul.neuron import functional
from soul.utils import multi_time_forward


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(self, lif, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.lif1 = deepcopy(lif)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.lif2 = deepcopy(lif)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.downsample_lif = deepcopy(lif)

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x): 
        # (T=1, B, N, W)
        out = multi_time_forward(x, [self.conv1, self.chomp1])
        out = self.lif1(out)

        out = multi_time_forward(out, [self.conv2, self.chomp2])
        out = self.lif2(out)

        if self.downsample:
            res = multi_time_forward(x, self.downsample)
            res = self.downsample_lif(res)
        else:
            res = x

        return out + res    

class TemporalConvNet(nn.Module):
    def __init__(self, lif, num_inputs, num_channels, kernel_size=2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                lif,
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                dilation=dilation_size, 
                padding=(kernel_size-1) * dilation_size, 
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # (T=1, B, N, W)
        return self.network(x) 


class SpikingTCN(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = config['input_channels']
        self.T = 1 # for sequential modeling, the time step is fixed to 1, and we use the sequence level to modeling

        lif = config['neuron']
        num_classes = config['num_classes']

        self.tcn = TemporalConvNet(lif, in_channels, config['hidden_channels'], kernel_size=config['ksize'])

        self.head = nn.Linear(config['hidden_channels'][-1], num_classes)

    def init_weights(self):
        self.head.weight.data.normal_(0, 0.01)

    def forward(self, x):
        functional.reset_net(self)
        # (1, B, N, W)
        y1 = self.tcn(x).mean(0) # -> (B, N, W)
        y1 = y1.permute(2, 0, 1).contiguous() # -> (W, B, N)
        output = self.head(y1).mean(0) # -> (B, num_classes)
        
        return output