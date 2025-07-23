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
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy

from soul.neuron import functional
from soul.utils.surrogate import Erf, ATan

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

def directional_rnn_cell_forward(cell: nn.Module, x: torch.Tensor,
                                   states: torch.Tensor):

    T = x.shape[0]
    ss = states

    output = []
    for t in range(T):
        ss = cell(x[t], ss)
        if states.dim() == 2:
            output.append(ss)
        elif states.dim() == 3:
            output.append(ss[0])
            # 当RNN cell具有多个隐藏状态时，通常第0个隐藏状态是其输出
    return torch.stack(output), ss

def bidirectional_rnn_cell_forward(cell: nn.Module, cell_reverse: nn.Module, x: torch.Tensor,
                                   states: torch.Tensor, states_reverse: torch.Tensor):
    T = x.shape[0]
    ss = states
    ss_r = states_reverse
    output = []
    output_r = []
    for t in range(T):
        ss = cell(x[t], ss)
        ss_r = cell_reverse(x[T - t - 1], ss_r)
        if states.dim() == 2:
            output.append(ss)
            output_r.append(ss_r)
        elif states.dim() == 3:
            output.append(ss[0])
            output_r.append(ss_r[0])
            # 当RNN cell具有多个隐藏状态时，通常第0个隐藏状态是其输出

    ret = []
    for t in range(T):
        ret.append(torch.cat((output[t], output_r[T - t - 1]), dim=-1))
    return torch.stack(ret), ss, ss_r

class SpikingRNNCellBase(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def weight_ih(self):
        return self.linear_ih.weight

    def weight_hh(self):

        return self.linear_hh.weight

    def bias_ih(self):

        return self.linear_ih.bias

    def bias_hh(self):

        return self.linear_hh.bias

class SpikingRNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False, *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout_p = dropout_p
        self.invariant_dropout_mask = invariant_dropout_mask
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.cells, self.cells_reverse = self.create_cells(*args, **kwargs)

        else:
            self.cells = self.create_cells(*args, **kwargs)

    def create_cells(self, *args, **kwargs):
        if self.bidirectional:
            cells = []
            cells_reverse = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            cells_reverse.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
                cells_reverse.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells), nn.Sequential(*cells_reverse)

        else:
            cells = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        raise NotImplementedError

    def forward(self, x: torch.Tensor, states=None):
        # x.shape=[T, batch_size, input_size]
        # states states_num 个 [num_layers * num_directions, batch, hidden_size]
        T = x.shape[0]
        batch_size = x.shape[1]

        if isinstance(states, tuple):
            # states非None且为tuple，则合并成tensor
            states_list = torch.stack(states)
            # shape = [self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]
        elif isinstance(states, torch.Tensor):
            if states.dim() == 3:
                states_list = states
            else:
                raise TypeError
        elif states == None:
            if self.bidirectional == True:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers*2, x.shape[1], self.hidden_size], dtype=torch.float, device=x.device).squeeze(0)
            else:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers, x.shape[1], self.hidden_size], dtype=torch.float, device=x.device).squeeze(0)
            
        else:
            raise TypeError

        if self.bidirectional:
            # 判断 num_direction*num_layers 是否符合要求，否则 new_states_list 会存在额外的0矩阵
            if (states_list.dim() == 4 and states_list.shape[1] != 2*self.num_layers) or (states_list.dim() == 3 and states_list.shape[0] != 2*self.num_layers):
                raise ValueError
            # y 表示第i层的输出。初始化时，y即为输入
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)
            for i in range(self.num_layers):
                # 第i层神经元的起始状态从输入states_list获取
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                    cell_init_states_reverse = states_list[i + self.num_layers]
                else:
                    cell_init_states = states_list[:, i]
                    cell_init_states_reverse = states_list[:, i + self.num_layers]

                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss, ss_r = bidirectional_rnn_cell_forward(
                    self.cells[i], self.cells_reverse[i], y, cell_init_states, cell_init_states_reverse)
                # 更新states_list[i]
                if self.states_num() == 1:
                    new_states_list[i] = ss
                    new_states_list[i + self.num_layers] = ss_r
                else:
                    new_states_list[:, i] = torch.stack(ss)
                    new_states_list[:, i + self.num_layers] = torch.stack(ss_r)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                return y, tuple(new_states_list)
        
        else:
            # 判断 num_direction*num_layers 是否符合要求，否则 new_states_list 会存在额外的0矩阵
            if (states_list.dim() == 4 and states_list.shape[1] != self.num_layers) or (states_list.dim() == 3 and states_list.shape[0] != self.num_layers):
                raise ValueError
            # y 表示第i层的输出。初始化时，y即为输入
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)
            for i in range(self.num_layers):
                # 第i层神经元的起始状态从输入states_list获取
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                else:
                    cell_init_states = states_list[:, i]

                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss = directional_rnn_cell_forward(
                    self.cells[i], y, cell_init_states)
                # 更新states_list[i]
                if self.states_num() == 1:
                    new_states_list[i] = ss
                else:
                    new_states_list[:, i] = torch.stack(ss)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                return y, tuple(new_states_list)


class SpikingLSTMCell(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 surrogate_function1=Erf(), surrogate_function2=ATan()):
        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.reset_parameters()

    def forward(self, x: torch.Tensor, hc=None):
        if hc is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
            c = torch.zeros_like(h)
        else:
            h = hc[0]
            c = hc[1]

        if self.surrogate_function2 is None:
            i, f, g, o = torch.split(self.surrogate_function1(self.linear_ih(x) + self.linear_hh(h)),
                                     self.hidden_size, dim=1)
        else:
            i, f, g, o = torch.split(self.linear_ih(x) + self.linear_hh(h), self.hidden_size, dim=1)
            i = self.surrogate_function1(i)
            f = self.surrogate_function1(f)
            g = self.surrogate_function2(g)
            o = self.surrogate_function1(o)

        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking


        c = c * f + i * g

        with torch.no_grad():
            torch.clamp_max_(c, 1.)

        h = c * o
            
        return h, c

class SpikingLSTM(SpikingRNNBase):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False,
                 surrogate_function1=Erf(), surrogate_function2=ATan()):
        super().__init__(input_size, hidden_size, num_layers, bias, dropout_p, invariant_dropout_mask, bidirectional,
                         surrogate_function1, surrogate_function2)
    @staticmethod
    def base_cell():
        return SpikingLSTMCell

class DCL(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        in_channels = config['input_channels'] # number of sensor channels
        input_dim = config['input_dim']
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

        self.lstm = SpikingLSTM(n_channels * in_channels * input_dim, hidden_dim, num_layers=2, bias=False)

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
        # use spike lstm
        x = x.flatten(2) #  -> (T, B, L * C * D)
        x, _ = self.lstm(x) # -> (T, B, D)
        x = self.head(x.mean(0))
        return x  # (B, num_classes)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x
