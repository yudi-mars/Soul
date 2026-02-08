'''
Filename:
    motion_dataset.py

Author:
    Di Yu <yudi2023@zju.edu.cn>

Date Created:
    2025-07-10

Description:
    Load data from motion sensor. For those data without preprocessing, 
    we refer to the processing method of UCI HAR and samling in fixed-width 
    sliding windows with 128 readings/window and 50% overlap.

References:
    - Malekzadeh, M. et al., "Mobile Sensor Data Anonymization", IoTDI'2019.
    https://github.com/mmalekzadeh/motion-sense
    - Riccardo Presotto et al., "Combining Public Human Activity Recognition Datasets to Mitigate Labeled Data Scarcity", SMARTCOMP'2023
    https://github.com/getalp/SmartComp2023-HAR-Supervised-Pretraining
'''
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms  

from soul.utils.coding import coding_map
from . import register_dataset

class AddGaussianNoise:
    def __init__(self, mu=0., sigma=0.02): 
        self.mu, self.sigma = mu, sigma

    def __call__(self, x):
        return x + np.random.normal(self.mu, self.sigma, size=x.shape)
    
class TimeWarp:
    def __init__(self, max_warp=0.2): 
        self.max_warp = max_warp

    def __call__(self, x):
        tt = np.linspace(0, 1, x.shape[1])
        random_points = tt + np.random.normal(0, self.max_warp, size=tt.shape)
        random_points = np.clip(random_points, 0, 1)
        warped = np.zeros_like(x)
        for c in range(x.shape[0]):
            warped[c] = np.interp(tt, random_points, x[c])
        return warped
    
class Scaling:
    def __init__(self, sigma=0.1): 
        self.sigma = sigma

    def __call__(self, x):
        factor = np.random.normal(1.0, self.sigma, (x.shape[0], 1))
        return x * factor

class Resample:
    '''
    random up/down sampling
    '''
    def __init__(self, up=True, factor_range=(0.8,1.2)):
        self.up = up; self.fr = factor_range
    def __call__(self, x):
        factor = random.uniform(*self.fr)
        L = x.shape[1]
        new_L = int(np.clip(L * factor, 1, 2 * L))
        res = np.zeros((x.shape[0], new_L))
        for c in range(x.shape[0]):
            res[c] = np.interp(np.linspace(0, 1, new_L),
                               np.linspace(0, 1, L), x[c])
        # pad or slice to original length
        if new_L > L:
            res = res[:, :L]
        else:
            pad = np.zeros_like(x)
            pad[:, :new_L] = res
            res = pad
        return res

class Standardize:
    def __init__(self, mean=0.0, std=1.0):
        """
        mean: shape (channels,) 或 (channels,1)
        std: shape (channels,) 或 (channels,1)
        """
        self.mean = np.array(mean).reshape(-1, 1)
        self.std = np.array(std).reshape(-1, 1)
        # avoid zero division
        self.std[self.std == 0] = 1.0

    def __call__(self, x):
        # x shape: (channels, time_steps) for each input
        return (x - self.mean) / self.std

class MotionData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []

    input_shape = (None, None)
    num_classes = None

    def __init__(self, data_dir, coding_schema, time_step, window_size, step_size, seed=2025):
        self.data_dir = data_dir
        self.T = time_step
        self.encode = coding_schema
        self.seed = seed

        self.window_size = window_size
        self.step_size = step_size

    def _load_segments(self):
        """
        加载特定部分数据

        Returns:
            数据数组，目标数组
        """
        raise NotImplementedError

    def download_data(self):
        """
        下载/加载数据集

        Returns:
            None
        """
        raise NotImplementedError
    
    def get_dataset(self, train=True):
        """
        获取Dataset对象

        Args:
            train: 是否是训练集

        Returns:
            Dataset对象
        """
        class DummyDataset(Dataset):
            def __init__(self, data, targets, trsf, encode, time_steps):
                self.data = data
                self.targets = targets

                self.trsf = trsf
                self.encode = encode
                self.time_steps = time_steps
            
            def __getitem__(self, index):
                inputs = self.data[index]

                if self.trsf:
                    inputs = self.trsf(inputs)

                inputs = torch.tensor(inputs, dtype=torch.float32)

                # coding (C, D) -> (T, C, D)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            train_trsf = transforms.Compose([*self.train_trsf, *self.common_trsf])
            ds = DummyDataset(self.train_data, self.train_targets, train_trsf, self.encode, self.T)
        else:
            test_trsf = transforms.Compose([*self.test_trsf, *self.common_trsf])
            ds = DummyDataset(self.test_data, self.test_targets, test_trsf, self.encode, self.T)

        return ds
    
@register_dataset('ucihar')
class iUCIHAR(MotionData):
    def __init__(self, data_dir, coding_schema, time_step, window_size, step_size, seed):
        super().__init__(data_dir, coding_schema, time_step, window_size, step_size, seed)

        self.num_classes = 6
        self.input_shape = (9, 128) # this is fixed for uci har

    def _load_segments(self, signal_type='train'):
        SIGNALS = [
            'body_acc_x', 'body_acc_y', 'body_acc_z',
            'body_gyro_x', 'body_gyro_y', 'body_gyro_z',
            'total_acc_x', 'total_acc_y', 'total_acc_z'
        ]

        X_list = []
        for signal in SIGNALS:
            path = os.path.join(self.data_dir, signal_type, 'Inertial Signals', f"{signal}_{signal_type}.txt")
            arr = np.loadtxt(path)  # [N, 128]
            X_list.append(arr[:, np.newaxis, :])  # → [N,1,128]
        X = np.concatenate(X_list, axis=1)  # → [N, 9, 128]
        y = np.loadtxt(os.path.join(self.data_dir, signal_type, "y_" + signal_type + ".txt")).astype(int) - 1

        return X.astype(np.float32), y.astype(int)
    
    def download_data(self):
        # self.data_dir = ~/data/ucihar/

        # load train data
        self.train_data, self.train_targets = self._load_segments('train')
        # load test data
        self.test_data, self.test_targets = self._load_segments('test')

        self.train_mean = self.train_data.mean(axis=(0, 2))  # shape [C,]
        self.train_std  = self.train_data.std(axis=(0, 2))

        # to tensor
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long) # (B)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

        print(f'train data shape: {self.train_data.shape}, test data shape: {self.test_data.shape}')

    def get_dataset(self, train=True):
        self.train_trsf = [
            # Resample(factor_range=(0.9, 1.1)), 
            AddGaussianNoise(0., sigma=0.02),
            # TimeWarp(max_warp=0.2),
            Scaling(sigma=0.1),
        ]

        self.common_trsf = [
            Standardize(mean=self.train_mean, std=self.train_std),
        ]

        return super().get_dataset(train)

@register_dataset('motionsense') 
class iMotionSense(MotionData):
    def __init__(self, data_dir, coding_schema, time_step, window_size, step_size, seed):
        super().__init__(data_dir, coding_schema, time_step, window_size, step_size, seed)

        self.num_classes = 6
        self.input_shape = (12, self.window_size)

    def _load_segments(self, df, window_size=250, step_size=125):
        cols = [c for c in df.columns if c not in ('activity','subject','Unnamed: 0')]
        X, y = [], []
        for subj in df['subject'].unique():
            sub = df[df['subject'] == subj]
            for act in sub['activity'].unique():
                seg = sub[sub['activity']==act][cols].values
                
                for i in range(0, len(seg) - window_size + 1, step_size):
                    X.append(seg[i:i + window_size].transpose(0, 1))
                    y.append(act)
        return np.array(X), np.array(y)
    
    def download_data(self):
        '''
        the A_DeviceMotiondata is mostly recommended to be used
        MotionSense/
        ├── A_DeviceMotion_data/
        │   ├── dws1
        │   ├── dws2
        │   ├── ...
        │   └── wlk_15
        ├── data_subjects_info.csv
        └── ...
        '''
        # self.data_dir = ~/data/MotionSense/
        # load data
        data = []
        for trial_folder in os.listdir(os.path.join(self.data_dir, 'A_DeviceMotion_data')):
            trial_path = os.path.join(self.data_dir, "A_DeviceMotion_data", trial_folder)
            if os.path.isdir(trial_path):
                act = trial_folder # e.g., wlk_7
                # print(f'Processing activity: {act}...')
                for fname in os.listdir(trial_path):
                    if fname.endswith(".csv"):
                        # print(f'Processing {fname}...')
                        df = pd.read_csv(os.path.join(trial_path, fname))
                        df['activity'] = act.split('_')[0]
                        df['subject'] = fname.replace('.csv','')
                        data.append(df)
        df = pd.concat(data, ignore_index=True)

        X, y = self._load_segments(df, self.window_size, self.step_size)
        X = X.swapaxes(1, 2) # (B, D, C) -> (B, C, D)

        # categorize label
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # load train & test data
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        self.train_mean = self.train_data.mean(axis=(0, 2))  # shape [C,]
        self.train_std  = self.train_data.std(axis=(0, 2))

        # to tensor 
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long) # (B)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

        print(f'train data shape: {self.train_data.shape}, test data shape: {self.test_data.shape}')

    def get_dataset(self, train=True):
        self.common_trsf = [
            Standardize(mean=self.train_mean, std=self.train_std),
        ]

        return super().get_dataset(train)
    
@register_dataset('shoaib') 
class iShoaib(MotionData):
    def __init__(self, data_dir, coding_schema, time_step, window_size, step_size, seed):
        super().__init__(data_dir, coding_schema, time_step, window_size, step_size, seed)

        self.num_classes = 8
        self.input_shape = (12, self.window_size)

    def _load_segments(self, fname, window_size=250, step=125):
        segments = []
        labels = []
        feature_cols = [
            'Ax', 'Ay', 'Az', 
            'Lx', 'Ly', 'Lz', 
            'Gx', 'Gy', 'Gz', 
            'Mx', 'My', 'Mz', 
        ]
        # for each participant
        df = pd.read_csv(os.path.join(self.data_dir, fname), header=[0, 1])
        # only the last column contains corresponding label for each line
        activities = df.values[:, -1]
        # 5 position, then divided the original datafrom into 5 pieces
        n_splits = 5 
        cols_per_split = df.shape[1] // n_splits
        dfs = [df.iloc[:, i * cols_per_split:(i + 1) * cols_per_split] for i in range(n_splits)]

        # for data at each position
        for d in dfs: 
            d.columns = ['timestamp'] + feature_cols + ['activity']
            d['activity'] = activities

            # for each activity
            for activity, grp in d.groupby('activity'):
                X = grp[feature_cols].values
                for start in range(0, len(grp) - window_size + 1, step):
                    seg = X[start:start + window_size]
                    segments.append(seg) # one specific segment 
                    labels.append(activity)

        X_windows = np.array(segments).swapaxes(1, 2) # (B, D, C) -> (B, C, D)
        y_windows = np.array(labels) # (B)

        return X_windows, y_windows

    def download_data(self):
        '''
        after unzip the rar file, the shoaib data structure should be:
        shoaib/
        ├── Participant_1.csv
        ├── Participant_2.csv
        ├── ...
        ├── Participant_10.csv
        └── readme.txt
        '''
        # self.data_dir = ~/data/shoaib/
        X, y = [], []
        for fname in os.listdir(self.data_dir):
            if fname.endswith('.csv'):
                print(f'Processing {fname}...')
                # slice signal
                participant_X, participant_y = self._load_segments(fname, self.window_size, self.step_size)
                X.append(participant_X)
                y.append(participant_y)

        # summarize all data
        X = np.concatenate(X, axis=0) # (B, C, D)
        y = np.concatenate(y, axis=0) # (B)

        # categorize label y
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # load train & test data
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        self.train_mean = self.train_data.mean(axis=(0, 2))  # shape [C,]
        self.train_std  = self.train_data.std(axis=(0, 2))

        # to tensor
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long) # (B)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

        print(f'train data shape: {self.train_data.shape}, test data shape: {self.test_data.shape}')

    def get_dataset(self, train=True):
        self.common_trsf = [
            Standardize(mean=self.train_mean, std=self.train_std),
        ]

        return super().get_dataset(train)
    
@register_dataset('hhar') 
class iHHAR(MotionData):
    def __init__(self, data_dir, coding_schema, time_step, window_size, step_size, seed):
        super().__init__(data_dir, coding_schema, time_step, window_size, step_size, seed)

        self.num_classes = 6
        self.input_shape = (3, self.window_size)

    def _segment_signals(self, df, window_size=200, step=100):
        segments, labels = [], []
        for user in df['user'].unique():
            temp = df[df['user'] == user]
            for activity, grp in temp.groupby('activity'):
                X = grp[['ax', 'ay', 'az']].values.astype(np.float32)
                for start in range(0, len(grp) - window_size + 1, step):
                    seg = X[start:start + window_size]
                    segments.append(seg) # one specific segment 
                    labels.append(activity)

        X_windows = np.array(segments).swapaxes(1, 2) # (B, D, C) -> (B, C, D)
        y_windows = np.array(labels) # (B)

        return X_windows, y_windows
    
    def download_data(self):
        '''
        we only implement the Accelerometer data collected by phone as an example, the other data in HHAR can be processed in a similar way

        hhar/
        ├── readme.txt
        └── Phone_accelerometer.txt
        '''
        # self.data_dir = ~/data/hhar/

        # load data collected by phone-accelerometer (As an example)
        df = pd.read_csv(os.path.join(self.data_dir, 'Phones_accelerometer.csv'))
        print('finish loading HHAR phone accelerometer file...')
        df = df[['Creation_Time', 'x', 'y', 'z', 'User', 'gt']].copy()
        df.columns = ['timestamp', 'ax', 'ay', 'az', 'user', 'activity']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by=['user', 'activity', 'timestamp'], ascending=[True, True, False], inplace=True)
        print('finish sorting...')

        # segment data
        X, y = self._segment_signals(df, self.window_size, self.step_size)
        print('finish buiding sequences...')

        # categorize label
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # load train & test data
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        self.train_mean = self.train_data.mean(axis=(0, 2))  # shape [C,]
        self.train_std  = self.train_data.std(axis=(0, 2))

        # to tensor 
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long) # (B)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

        print(f'train data shape: {self.train_data.shape}, test data shape: {self.test_data.shape}')

    def get_dataset(self, train=True):
        self.common_trsf = [
            Standardize(mean=self.train_mean, std=self.train_std),
        ]

        return super().get_dataset(train)
