'''
Filename: acoustic_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-30
Description:
    Load data from acoustic sensor. For those audio file without preprocessing, 
    we choose mel-frequency spectrogram to convert, shape as [window size, num channels].
    For the dynamic acoustic sensing (DVS) datasets, we process them similarly into [window size, num channels].

References:
    - Hanle Zheng et al. "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics." Nature Communications 2023.
    - Xinyi Chen et al. "Neuromorphic Sequential Arena: A Benchmark for Neuromorphic Temporal Processing." IJCAI'2025.
    https://github.com/liyc5929/neuroseqbench
'''
import os
import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from soul.utils.coding import coding_map
from . import register_dataset

class AudioData:
    train_trsf = []
    test_trsf = []
    common_trsf = []

    input_shape = (None, None)
    num_classes = None

    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        self.data_dir = data_dir
        self.T = time_step
        self.encode = coding_schema
        self.seed = seed

        self.sr = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    def download_data(self):
        raise NotImplementedError
    
    def get_dataset(self, train=True):
        raise NotImplementedError

@register_dataset('gtzan')
class iGTZAN(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed)

    def download_data(self):

        return 
    
    def get_dataset(self, train=True):
    
        return     

    
@register_dataset('urbansound')
class iUrbanSound8K(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed)

    def download_data(self):

        return 
    
    def get_dataset(self, train=True):
    
        return 
    
@register_dataset('gsc')
class iGoogleSpeechCommands(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed)

    def download_data(self):
        # V2 data as default

        return 
    
    def get_dataset(self, train=True):
    
        return 
    

@register_dataset('shd')
class iSpikingHeidelbergDigits(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed)

        self.duration = 100 # window size
        self.n_mfcc = 700 # fixed channel number

        self.num_classes = 20
        self.input_shape = (self.duration, self.n_mfcc) # (window_size, channels)

    def _preprocess(self, times, units, label):
        data_label = torch.tensor(label, dtype=torch.int64)
        max_unit   = self.n_mfcc
        max_time   = 1
        dt         = 1 / self.duration # 
        time_frames  = int(max_time / dt) # the total samping frames
        list_input = []
        for i in range(time_frames):
            indexs = np.argwhere(times <= i * dt).flatten()
            vals   = units[indexs]; vals = vals[vals > 0]
            vector = np.zeros(max_unit); vector[max_unit - vals] = 1
            times  = np.delete(times, indexs)
            units  = np.delete(units, indexs)
            list_input.append(vector)
        data_input = torch.tensor(np.array(list_input), dtype=torch.float32)

        return data_input, data_label

    def download_data(self):
        # self.data_dir=/data/shd/
        '''
        The files below should be included in data_dir
        shd/
        ├── shd_test.h5
        └── shd_train.h5
        '''
        os.makedirs(os.path.join(self.data_dir, f'preprocessed_W{self.duration}'), exist_ok=True)
        for train_type in ['train', 'test']:
            preprocessed_data_root = os.path.join(self.data_dir, f'preprocessed_W{self.duration}', train_type)
            if os.path.exists(preprocessed_data_root): 
                # Data preloading
                print(f"The `saved_data_file` exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
                h5file = h5py.File(f"{preprocessed_data_root}/shd_preprocessed_data.h5", "r")
                input_iter = h5file["inputs"]
                label_iter = h5file["labels"]

                setattr(self, f'{train_type}_data', [])
                setattr(self, f'{train_type}_targets', [])

                for i in tqdm(range(len(label_iter)), ncols=80, desc=f'{train_type} preloading: '):
                    getattr(self, f'{train_type}_data').append(torch.tensor(input_iter[i], dtype=torch.float32))
                    getattr(self, f'{train_type}_targets').append(torch.tensor(label_iter[i], dtype=torch.int64))

            else:
                # Fetching original data
                h5file = h5py.File(f"{self.data_dir}/shd_{train_type}.h5", "r")
                times_iter = h5file["spikes"]["times"]
                units_iter = h5file["spikes"]["units"] 
                label_iter = h5file["labels"]

                # Data preloading and preprocessing
                setattr(self, f'{train_type}_data', [])
                setattr(self, f'{train_type}_targets', [])

                os.mkdir(preprocessed_data_root)
                print(f"The preprocessing of `{self.__class__.__name__}` start.")
                for i in tqdm(range(len(label_iter)), ncols=80, desc=f'{train_type} preprocessing: '):
                    times = times_iter[i]
                    units = units_iter[i]
                    label = label_iter[i]

                    data_input, data_label = self._preprocess(times, units, label)
                    
                    getattr(self, f'{train_type}_data').append(data_input)
                    getattr(self, f'{train_type}_targets').append(data_label)


                # Data saving
                print(f"The saving to path `{preprocessed_data_root}` start.")
                with h5py.File(f"{preprocessed_data_root}/shd_preprocessed_data.h5", "w") as fp:
                    saved_inputs = fp.create_dataset("inputs", (len(getattr(self, f'{train_type}_data')), *getattr(self, f'{train_type}_data')[0].shape), dtype=np.float32)
                    saved_labels = fp.create_dataset("labels", (len(getattr(self, f'{train_type}_targets')), *getattr(self, f'{train_type}_targets')[0].shape), dtype=np.int64)

                    for i in tqdm(range(len(getattr(self, f'{train_type}_targets'))), ncols=80, desc=f'{train_type} saving: '):
                        saved_inputs[i] = getattr(self, f'{train_type}_data')[i].cpu()
                        saved_labels[i] = getattr(self, f'{train_type}_targets')[i].cpu()

    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                # (C, W=700)
                inputs = self.data[index]

                # coding (C, W) -> (inner_T, C, W)
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
        
@register_dataset('ssc')
class iSpikingSpeechCommands(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed)

        self.duration = 100
        self.n_mfcc = 700 # channel number

        self.num_classes = 35
        self.input_shape = (self.duration, self.n_mfcc) # (window_size, channels)

    def _preprocess(self, times, units, label):
        data_label = torch.tensor(label, dtype=torch.int64)
        max_unit   = self.n_mfcc # this is the channel number
        max_time   = 1
        dt         = 1 / self.duration
        time_step  = int(max_time / dt)
        list_input = []
        for i in range(time_step):
            indexs = np.argwhere(times <= i * dt).flatten()
            vals   = units[indexs]; vals = vals[vals > 0]
            vector = np.zeros(max_unit); vector[max_unit - vals] = 1
            times  = np.delete(times, indexs)
            units  = np.delete(units, indexs)
            list_input.append(vector)
        data_input = torch.tensor(np.array(list_input), dtype=torch.float32)
        return data_input, data_label
    
    def download_data(self):
        os.makedirs(os.path.join(self.data_dir, f'preprocessed_W{self.duration}'), exist_ok=True)
        for train_type in ['train', 'test']:
            preprocessed_data_root = os.path.join(self.data_dir, f'preprocessed_W{self.duration}', train_type)
            if os.path.exists(preprocessed_data_root):
                # Data preloading
                print(f"The `saved_data_file` exists, data preloading of `{self.__class__.__name__}` from path `{preprocessed_data_root}` start.")
                h5file = h5py.File(f"{preprocessed_data_root}/ssc_preprocessed_data.h5", "r")
                input_iter = h5file["inputs"]
                label_iter = h5file["labels"]

                setattr(self, f'{train_type}_data', [])
                setattr(self, f'{train_type}_targets', [])
                for i in tqdm(range(len(label_iter)), ncols=80, desc=f'{train_type} preloading: '):
                    getattr(self, f'{train_type}_data').append(torch.tensor(input_iter[i], dtype=torch.float32))
                    getattr(self, f'{train_type}_targets').append(torch.tensor(label_iter[i], dtype=torch.int64))
            else:
                # Fetching original data
                h5file = h5py.File(f"{self.data_dir}/ssc_{train_type}.h5", "r")
                times_iter = h5file["spikes"]["times"]
                units_iter = h5file["spikes"]["units"] 
                label_iter = h5file["labels"]

                # Data preprocessing
                setattr(self, f'{train_type}_data', [])
                setattr(self, f'{train_type}_targets', [])

                os.mkdir(preprocessed_data_root)
                print(f"The preprocessing of `{self.__class__.__name__}` start.")
                for i in tqdm(range(len(label_iter)), ncols=80, desc=f'{train_type} preprocessing: '):
                    times = times_iter[i]
                    units = units_iter[i]
                    label = label_iter[i]

                    data_input, data_label = self._preprocess(times, units, label)

                    getattr(self, f'{train_type}_data').append(data_input)
                    getattr(self, f'{train_type}_targets').append(data_label)

                # Data saving
                print(f"The saving to path `{preprocessed_data_root}` start.")
                with h5py.File(f"{preprocessed_data_root}/shd_preprocessed_data.h5", "w") as fp:
                    saved_inputs = fp.create_dataset("inputs", (len(getattr(self, f'{train_type}_data')), *getattr(self, f'{train_type}_data')[0].shape), dtype=np.float32)
                    saved_labels = fp.create_dataset("labels", (len(getattr(self, f'{train_type}_targets')), *getattr(self, f'{train_type}_targets')[0].shape), dtype=np.int64)

                    for i in tqdm(range(len(getattr(self, f'{train_type}_targets'))), ncols=80, desc=f'{train_type} saving:'):
                        saved_inputs[i] = getattr(self, f'{train_type}_data')[i].cpu()
                        saved_labels[i] = getattr(self, f'{train_type}_targets')[i].cpu()

    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                # (W, C) -> (C, W)
                inputs = self.data[index].transpose(0, 1)

                # coding (C, W) -> (inner_T, C, W)
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
