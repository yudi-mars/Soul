"""
Filename: wireless_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-08-07
Description:
    Load data collected by wireless sensor. 

References:
    - Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.
    https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
"""
import os
import glob
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import Dataset

from soul.utils.coding import coding_map
from . import register_dataset


class WirelessData:
    input_shape = (None, None, None) # Channel State Information (CSI) size
    num_classes = None

    def __init__(self, data_dir, coding_schema, time_step):
        self.data_dir = data_dir
        self.T = time_step
        self.encode = coding_schema

    def download_data(self):
        raise NotImplementedError
    
    def get_dataset(self, train=True):
        raise NotImplementedError
    
@register_dataset('widar')
class iWidar3(WirelessData):

    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 22
        self.input_shape = (22, 20, 20)

    def download_data(self):
        '''
        Widardata/
        ├── test
        └── train
        '''
        # self.data_dir = /data/Widardata/

        # load train data
        self.train_data = glob.glob(self.data_dir + 'train/*/*.csv')
        folder = glob.glob(self.data_dir + 'train/*/')
        category = {folder[i].split('/')[-2] : i for i in range(len(folder))}
        self.train_targets = [category[f.split('/')[-2]]  for f in self.train_data]

        # load test data
        self.test_data = glob.glob(self.data_dir + 'test/*/*.csv')
        folder = glob.glob(self.data_dir + 'test/*/')
        self.test_targets = [category[f.split('/')[-2]]  for f in self.test_data]
    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                inputs = np.genfromtxt(self.data[index], delimiter=',')  

                # normalize 
                inputs = (inputs - 0.0025) / 0.0119

                # reshape: (22, 400) -> (22, 20, 20), maybe can interpolate to 32*32?
                inputs = inputs.reshape(22, 20, 20)

                # to tensor
                inputs = torch.tensor(inputs, dtype=torch.float32)

                # coding (C, H, W) -> (T, C, H, W)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(self.train_data, self.train_targets, self.encode, self.T)
        else:
            ds = DummyDataset(self.test_data, self.test_targets, self.encode, self.T)

        return ds

@register_dataset('uthar') 
class iUTHAR(WirelessData):
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 7
        self.input_shape = (1, 250, 90)

    def download_data(self):
        '''
        UT_HAR/
        ├── data
        └── label
        '''
        # self.data_dir = /data/UT_HAR/

        data_list = glob.glob(self.data_dir + '/data/*.csv')
        label_list = glob.glob(self.data_dir + '/label/*.csv')
        WiFi_data = {}
        for data_dir in data_list:
            data_name = data_dir.split('/')[-1].split('.')[0]
            with open(data_dir, 'rb') as f:
                data = np.load(f)
                data = data.reshape(len(data), 1, 250, 90)
                data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))

            WiFi_data[data_name] = torch.tensor(data_norm, dtype=torch.float32)

        for label_dir in label_list:
            label_name = label_dir.split('/')[-1].split('.')[0]
            with open(label_dir, 'rb') as f:
                label = np.load(f)
            WiFi_data[label_name] = torch.tensor(label, dtype=torch.long)


        self.train_data = WiFi_data['X_train']
        self.train_targets = WiFi_data['y_train']

        self.test_data = torch.cat((WiFi_data['X_val'],WiFi_data['X_test']), 0)
        self.test_targets = torch.cat((WiFi_data['y_val'], WiFi_data['y_test']), 0)
    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                inputs = self.data[index]

                # coding (C, H, W) -> (T, C, H, W)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(self.train_data, self.train_targets, self.encode, self.T)
        else:
            ds = DummyDataset(self.test_data, self.test_targets, self.encode, self.T)

        return ds
        
    
@register_dataset('fihumanid')
class iFiHumanID(WirelessData):
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 14
        self.input_shape = (3, 114, 500)

    def download_data(self):
        '''
        NTU-Fi-HumanID/
        ├── test_amp
        └── train_amp
        '''
        # self.data_dir = ./data/NTU-Fi-HumanID

        # load train data
        fpath = os.path.join(self.data_dir, 'train_amp')
        self.train_data = glob.glob(fpath + '/*/*.mat')
        folder = glob.glob(fpath + '/*/')

        category = {folder[i].split('/')[-2] : i for i in range(len(folder))}

        self.train_targets = [category[mat.split('/')[-2]] for mat in self.train_data]

        # load test data
        fpath = os.path.join(self.data_dir, 'test_amp')
        self.test_data = glob.glob(fpath + '/*/*.mat')
        folder = glob.glob(fpath + '/*/')

        self.test_targets = [category[mat.split('/')[-2]] for mat in self.test_data]

    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                inputs = sio.loadmat(self.data[index])['CSIamp']
                inputs = (inputs - 42.3199) / 4.9802

                # sampling: 2000 -> 500
                inputs = inputs[:, ::4]
                inputs = inputs.reshape(3, 114, 500)

                # to tensor
                inputs = torch.tensor(inputs, dtype=torch.float32)

                # coding (C, H, W) -> (T, C, H, W)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(self.train_data, self.train_targets, self.encode, self.T)
        else:
            ds = DummyDataset(self.test_data, self.test_targets, self.encode, self.T)

        return ds
        
    
@register_dataset('fihar')
class iFiHAR(WirelessData):
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 6
        self.input_shape = (3, 114, 500)

    def download_data(self):
        '''
        NTU-Fi_HAR/
        ├── test_amp
        └── train_amp
        '''
        # self.data_dir = ./data/NTU-Fi_HAR/

        # load train data
        fpath = os.path.join(self.data_dir, 'train_amp')
        self.train_data = glob.glob(fpath + '/*/*.mat')
        folder = glob.glob(fpath + '/*/')

        category = {folder[i].split('/')[-2] : i for i in range(len(folder))}

        self.train_targets = [category[mat.split('/')[-2]] for mat in self.train_data]

        # load test data
        fpath = os.path.join(self.data_dir, 'test_amp')
        self.test_data = glob.glob(fpath + '/*/*.mat')
        folder = glob.glob(fpath + '/*/')

        self.test_targets = [category[mat.split('/')[-2]] for mat in self.test_data]
         
    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                inputs = sio.loadmat(self.data[index])['CSIamp']
                inputs = (inputs - 42.3199) / 4.9802

                # sampling: 2000 -> 500
                inputs = inputs[:, ::4]
                inputs = inputs.reshape(3, 114, 500)

                # to tensor
                inputs = torch.tensor(inputs, dtype=torch.float32)

                # coding (C, H, W) -> (T, C, H, W)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(self.train_data, self.train_targets, self.encode, self.T)
        else:
            ds = DummyDataset(self.test_data, self.test_targets, self.encode, self.T)

        return ds
