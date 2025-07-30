'''
Filename: acoustic_dataset.py
Author: Di Yu <yudi2023@zju.edu.cn>
Date Created: 2025-07-30
Description:
    Load data from acoustic sensor. For those audio file without preprocessing, 
    we choose mel-frequency spectrogram to convert.

References:
    - 
'''

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
    
@register_dataset('urbansound')
class iUrbanSound8K(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, sample_rate, duration, n_mfcc, hop_length, seed)

    
    