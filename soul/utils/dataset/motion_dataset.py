'''
Filename: motion_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-10
Description:
    Load data from motion sensor with some data augmentation operations

References:
    - Malekzadeh, M. et al, "Mobile Sensor Data Anonymization", 2019.
    https://github.com/mmalekzadeh/motion-sense
'''
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch        
    
class MotionData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []

    input_shape = None, None, None
    num_classes = None

    data_source = None

    def __init__(self, data_dir, T):
        self.data_dir = data_dir
        self.T = T

    def download_data(self):
        raise NotImplementedError

class iUCIHAR(MotionData):
    data_source = 'tensor'

    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

    def download_data(self):
        # self.data_dir = ~/data/ucihar/

        # load train data
        self.train_data = pd.read_csv(
            os.path.join(self.data_dir, 'train', 'X_train.txt'),
            delim_whitespace=True, 
            header=None
        ).values
        self.train_targets = pd.read_csv(
            os.path.join(self.data_dir, 'train', 'y_train.txt'),
            delim_whitespace=True, 
            header=None
        ).values.squeeze() - 1 # label start from 0

        # load test data
        self.test_data = pd.read_csv(
            os.path.join(self.data_dir, 'test', 'X_test.txt'),
            delim_whitespace=True, 
            header=None
        ).values
        self.test_targets = pd.read_csv(
            os.path.join(self.data_dir, 'test', 'y_test.txt'),
            delim_whitespace=True, 
            header=None
        ).values.squeeze() - 1 # label start from 0

        self.train_data = self.train_data.astype(np.float32) # (B, D) D=561 in this dataset
        self.test_data = self.test_data.astype(np.float32)
        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long) # (B)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

        mean = self.train_data.mean(axis=0) # (D, )
        std = self.train_data.std(axis=0) # (D, )

        self.train_data = torch.tensor(self.train_data, dtype=torch.float32)
        self.train_data = (self.train_data - mean) / (std + 1e-5)

        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)
        self.test_data = (self.test_data - mean) / (std + 1e-5)

        # (B, D) -> (B, 1, D) for window dimension
        self.train_data = self.train_data.unsqueeze(1) 
        self.test_data = self.test_data.unsqueeze(1) 

class iMotionSense(MotionData):
    data_source = 'tensor'

    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

    def _segment_signals(self, df, window_size=250, step_size=125):
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
                for fname in os.listdir(trial_path):
                    if fname.endswith(".csv"):
                        df = pd.read_csv(os.path.join(trial_path, fname))
                        df['activity'] = act.split('_')[0]
                        df['subject'] = fname.replace('.csv','')
                        data.append(df)
        df = pd.concat(data, ignore_index=True)

        X, y = self._segment_signals(df, 250, 125)
        X = X.swapaxes(1, 2) # (B, D, C) -> (B, C, D)

        # categorize label
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # load train & test data
        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=self.seed
        )

        # normalize
        scaler = StandardScaler()
        B, C, D = self.train_data.shape
        X_train_flat = self.train_data.reshape(B, C * D)
        X_train_scaled = scaler.fit_transform(X_train_flat)
        self.train_data = X_train_scaled.reshape(B, C, D)

        B, C, D = self.test_data.shape
        X_test_flat = self.test_data.reshape(B, C * D)
        X_test_scaled = scaler.transform(X_test_flat)
        self.test_data = X_test_scaled.reshape(B, C, D)

        # to tensor 
        self.train_data = torch.tensor(self.train_data, dtype=torch.float32) # (B, C, D)
        self.test_data = torch.tensor(self.test_data, dtype=torch.float32)

        self.train_targets = torch.tensor(self.train_targets, dtype=torch.long) # (B)
        self.test_targets = torch.tensor(self.test_targets, dtype=torch.long)

class iHHAR(MotionData):
    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

class iShoaib(MotionData):
    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

