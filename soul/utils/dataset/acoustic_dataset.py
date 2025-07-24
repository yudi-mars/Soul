'''
Filename: acoustic_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-23
Description:
    Load data from acoustic sensor. The sound file need to be proceessed into Mel spectrograms.

References:
    - 
'''
import os
import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torchaudio

class AcousticDataset(object):
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