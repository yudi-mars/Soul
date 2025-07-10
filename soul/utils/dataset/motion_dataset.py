'''
Filename: motion_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-10
Description:
    Load data from motion sensor with some data augmentation operations

References:
    - Yang, H. et al., "Empirical Evaluation of Data Augmentations for Biobehavioral Time Series Data with Deep Learning", 2022.
      https://github.com/comp-well-org/Data_Augmentation_for_Biobehavioral_Time_Series_Data
    - A le Guennec et al., "Data Augmentation for Time Series Classification using Convolutional Neural Networks", 2016
'''
import os
import random
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import torch

# ============================== Transformation Methods ==============================
class Standardize(object):
    def __init__(self, mean, std, eps=1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return (x - self.mean) / (self.std + self.eps)
    
class Negate(object):
    '''
    Flip signals to correct for sensor orientation offsets.
    '''
    def __call__(self, x):
        return -x
    
class Scale:
    '''
    Randomly scale signals to simulate variations in motion intensity
    '''
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, x):
        factor = torch.randn(1).item() * self.sigma + 1.0
        return x * factor
    
class RandomCropResize(object):
    ''' 
    Simulate variations in sampling rate and sequence length to handle dynamic frequencies and durations. 
    '''
    def __init__(self, scale=(0.8, 1.0)):
        self.scale = scale

    def __call__(self, x):
        L = x.size(0) # (D)
        newL = int(L * random.uniform(*self.scale))
        start = random.randint(0, L - newL)
        cropped = x[start:start + newL]
        return torch.nn.functional.interpolate(
            cropped.unsqueeze(0).unsqueeze(0),
            size=L, 
            mode='linear', 
            align_corners=False
        ).squeeze()
    
class AddNoise:
    '''
    Adding synthetic sensor noise to enhance robustness against real-world signal perturbations
    '''
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, x):
        return x + torch.randn_like(x) * self.sigma
    
class Permutation:
    '''
    Shuffling signal segments to increase temporal randomness and improve generalization.
    '''
    def __init__(self, n_segments=4):
        self.n_segments = n_segments

    def __call__(self, x):
        L = x.size(0)
        segs = torch.chunk(x, self.n_segments, dim=0)
        perm = random.sample(segs, len(segs))
        return torch.cat(perm, dim=0)
    
class TimeWarp:
    '''
    Use time warping to mimic speed variability in motion.
    '''
    def __init__(self, sigma=0.2, knot=4):
        self.sigma = sigma
        self.knot = knot

    def __call__(self, x):
        x_np = x.numpy()
        L = x_np.shape[0]
        xx = np.arange(L)
        random_warp = np.cumsum(
            np.random.randn(self.knot) * self.sigma
        )
        warp_steps = np.interp(xx, np.linspace(0, L-1, num=self.knot), random_warp)
        tt = xx + warp_steps
        cs = CubicSpline(xx, x_np)
        warped = cs(tt)
        return torch.from_numpy(warped).float()
    
# ============================== Customized Dataset ==============================
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
    data_source = 'npy'

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

        self.train_trsf = [
            Standardize(mean, std),
            # AddNoise(0.02),
            # Scale(0.1),
            # Permutation(4),
            # RandomCropResize((0.8, 1.0)),
        ]

        self.test_trsf = [
            Standardize(mean, std),
        ]

class iMotionSense(MotionData):
    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

class iHHAR(MotionData):
    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

class iShoaib(MotionData):
    def __init__(self, data_dir, T):
        super().__init__(data_dir, T)

