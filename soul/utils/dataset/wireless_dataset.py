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
import copy
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from soul.utils.coding import coding_map
from . import register_dataset


class WirelessData:
    input_shape = (None, None, None) # Channel State Information (CSI) size, (C, H, W), can reshape to (CH, W) for RNN input, CH is the sequential dimension
    num_classes = None

    def __init__(self, data_dir, coding_schema, time_step):
        self.data_dir = data_dir
        self.T = time_step
        self.encode = coding_schema

    def download_data(self):
        raise NotImplementedError
    
    def get_dataset(self, train=True):
        raise NotImplementedError
    
@register_dataset('aril')
class iARIL(WirelessData):
    '''
    We only use amplitude information for ARIL dataset as baseline.
    '''
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 6
        self.input_shape = (1, 52, 192) 

    def download_data(self):
        # self.data_dir = /data/ARIL/
        train_data = sio.loadmat(os.path.join(self.data_dir, 'train_data_split_amp.mat'))
        test_data = sio.loadmat(os.path.join(self.data_dir, 'test_data_split_amp.mat'))

        self.train_data = train_data['train_data'][:, np.newaxis, :, :]  # (N, 52, 192) -> (N, 1, 52, 192)
        self.train_targets = train_data['train_activity_label'].reshape(-1) # (N, 1) -> (N,)

        self.test_data = test_data['test_data'][:, np.newaxis, :, :]  # (N, 52, 192) -> (N, 1, 52, 192)
        self.test_targets = test_data['test_activity_label'].reshape(-1) # (N, 1) -> (N,)

        # to tensor
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long)
        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

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

@register_dataset('gaitid')
class iGaitID(WirelessData):
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 6
        self.input_shape = (6, 512, 21) # (A, S, T) 

    def download_data(self):
        # self.data_dir = /data/ElderAL/

        return 

    
    def get_dataset(self, train=True):
        return 
    
@register_dataset('falldar')
class iFallDar(WirelessData):
    '''
    The CSI data for Falldar, we use the denoise version data provided in this dataset for experiments.
    0: not fall
    1: fall
    '''
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 2 # fall / no-fall
        self.input_shape = (3, 32, 500) 

    def download_data(self):
        # self.data_dir = /data/falldar/
        data_dir = os.path.join(self.data_dir, 'denoisemat')

        samples, labels = [], []
        for code in ['fall', 'nonfall']:
            print(f'Processing {code} data...')
            folder_path = os.path.join(data_dir, code)
            for file_path in os.listdir(folder_path): 
                for data_mat_file in os.listdir(os.path.join(folder_path, file_path)):
                    if data_mat_file.endswith('.mat'):
                        mat_contents = sio.loadmat(os.path.join(folder_path, file_path, data_mat_file))
                        data = mat_contents['csi_data']  # shape: (2000, 30, 3) CSI complex data
                        data = np.swapaxes(data, 0, 2)  # (3, 30, 2000) : [attenna, subcarrier, time packet]
                        amp = np.abs(data)  # use magnitude only

                        # interpolate to 32 * 500
                        amp = F.interpolate(torch.from_numpy(amp).unsqueeze(0).float(), size=(32, 500), mode='bilinear', align_corners=False)
                        amp = amp.squeeze(0).numpy()

                        samples.append(amp)
                        labels.append(0 if code == 'nonfall' else 1)

        samples = np.array(samples) # (1655, 3, 30, 500) 
        labels = np.array(labels)

        # magnitude Z-score normalization
        mean = np.mean(samples, axis=(0, 2, 3), keepdims=True)  # shape (1, 3, 1, 1)
        std = np.std(samples, axis=(0, 2, 3), keepdims=True)    # shape (1, 3, 1, 1)
        samples = (samples - mean) / (std + 1e-6)

        # convert to tensor for encoding
        samples = torch.from_numpy(samples).float()

        # load train and test data
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            samples, labels, test_size=0.1, stratify=labels, random_state=2025
        )   

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
    
@register_dataset('wigesture')
class iWiGesture(WirelessData):
    '''
    The CSI data for WiGesture is preprocessed to gray-scale maginitude images with size 32x32.
    Each sample corresponds to a gesture action.
    There are 6 gesture classes in total.
    0: applause
    1: circle clockwise
    2: front and after
    3: left to right
    4: up and down
    5: wave right
    Only dynamic data is used in previous works.
    '''
    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 6 
        self.input_shape = (1, 32, 32) 

    def _handle_complex_data(self, x, valid_indices):
        real_parts = []
        imag_parts = []
        for i in valid_indices:
            real_parts.append(x[i * 2])
            imag_parts.append(x[i * 2 - 1])
        return np.array(real_parts) + 1j * np.array(imag_parts)

    def _get_time(self, s):
        s = s.split()[-1]
        s = s.split(':')
        h = float(s[0])
        m = float(s[1])
        t = float(s[2])
        total = h * 3600 + m * 60 + t
        return h, m, t, total

    def download_data(self):
        # self.data_dir = /data/wigesture/
        data_dir = os.path.join(self.data_dir, 'dynamic')
        
        # 52 subcarriers are valid
        csi_vaid_subcarrier_index = range(0, 52)
        csi = []

        for people in sorted(os.listdir(data_dir)):
            path_people = os.path.join(data_dir, people)

            for action_file in sorted(os.listdir(path_people)): # sort to keep the same order
                action = action_file.split('.')[0] # label.csv
                print(f'Processing {people} {action}...')
                file_path = os.path.join(path_people, action_file) # action file path    

                df = pd.read_csv(file_path)
                df.dropna(inplace=True)
                df['data'] = df['data'].apply(lambda x: eval(x))
                complex_data = df['data'].apply(lambda x: self._handle_complex_data(x, csi_vaid_subcarrier_index))
                magnitude = complex_data.apply(lambda x: np.abs(x))
                phase = complex_data.apply(lambda x: np.angle(x, deg=True))
                local_time = np.array(df['local_timestamp'])

                csi.append({
                    'csi_local_time': local_time,
                    'action': action,
                    'magnitude': np.array([np.array(a) for a in magnitude]),
                    'phase': np.array([np.array(a) for a in phase])
                })

        gap = 1
        length = gap * 100 # the time window length, can be adjusted here
        action_list = []
        magnitudes = []
        # phases = []

        for data in csi:
            # local_time = data['csi_local_time']
            magnitude = data['magnitude']
            # phase = data['phase']
            action = data['action']

            index=0
            while index < len(magnitude) - length:
                current_magnitude = magnitude[index:index + length]
                # current_phase = phase[index:index + length]
                index += (length + gap - 1)
                magnitudes.append(copy.deepcopy(current_magnitude))
                # phases.append(copy.deepcopy(current_phase))
                action_list.append(action)
        
        action_list = np.array(action_list)
        magnitudes = np.array(magnitudes)
        # phases=np.array(phases)
        magnitudes = np.swapaxes(magnitudes, 1, 2) # (N, L, 52) -> (N, 52, L)
        magnitudes = np.expand_dims(magnitudes, axis=1) # (N, 52, L) -> (N, 1, 52, L)
        # sometimes phases can be also introduced, but this data may fluctuate a lot due to different devices/environments

        # magnitude Z-score normalization
        magnitudes = (magnitudes - magnitudes.mean()) / (magnitudes.std() + 1e-6)

        # resize to better fit the model input
        magnitudes = F.interpolate(torch.tensor(magnitudes, dtype=torch.float32), size=(32, 32), mode='bilinear', align_corners=False) # (N, 1, 52, L) -> (N, 1, 32, 32)

        # catgorize labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(action_list)

        # load train and test data
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            magnitudes, y_encoded, test_size=0.1, stratify=y_encoded, random_state=2025
        )    

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
    
@register_dataset('widar')
class iWidar3(WirelessData):

    def __init__(self, data_dir, coding_schema, time_step):
        super().__init__(data_dir, coding_schema, time_step)

        self.num_classes = 22
        self.input_shape = (22, 32, 32)

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

                # reshape: (22, 400) -> (22, 20, 20)
                inputs = inputs.reshape(22, 20, 20)

                # to tensor
                inputs = torch.tensor(inputs, dtype=torch.float32)

                # interpolate to 32*32
                inputs = F.interpolate(inputs.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
                inputs = inputs.squeeze(0)

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
