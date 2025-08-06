"""
Filename: wireless_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-08-07
Description:
    Load data from wireless sensor. 

    NTU-Fi_HAR/
    ├── test_amp
    └── train_amp

    NTU-Fi-HumanID/
    ├── test_amp
    └── train_amp

    UT_HAR/
    ├── data
    └── label

    Widardata/
    ├── test
    └── train

References:
    - Jianfei Yang et al. "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing." Patterns'2023.
    https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from soul.utils.coding import coding_map
from . import register_dataset


class WirelessData:
    input_shape = (None, None)
    num_classes = None

    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        self.data_dir = data_dir
        self.T = time_step
        self.encode = coding_schema
        self.seed = seed

    def download_data(self):
        raise NotImplementedError
    
    def get_dataset(self, train=True):
        raise NotImplementedError