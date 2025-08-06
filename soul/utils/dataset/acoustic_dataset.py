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
import cv2
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from soul.utils.coding import coding_map
from . import register_dataset

class AudioData:
    input_shape = (None, None)
    num_classes = None

    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        self.data_dir = data_dir
        self.T = time_step
        self.encode = coding_schema
        self.seed = seed
        self.reduce_size = reduce_size

        self.sr = None              # (int) target sampling rate for audio data
        self.duration = None        # (int) only load up to this much audio (in seconds)
        self.n_mels = None          # (int) number of Mel bands to generate
        self.n_mfcc = None          # (int) number of MFCCs to return
        self.n_fft = None           # (int) length of the FFT window
        self.hop_length = None      # (int) number of samples between successive frames

        self.input_shape = (self.reduce_size, self.reduce_size)

    def download_data(self):
        raise NotImplementedError
    
    def get_dataset(self, train=True):
        raise NotImplementedError

@register_dataset('gtzan')
class iGTZAN(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, reduce_size, seed)

        self.hop_length = 256
        self.n_mels = 96
        self.n_fft = 512
        self.sr = 22050
        self.duration = 30

        self.num_classes = 10

    def download_data(self):
        # self.data_dir = /data/gtzan/
        '''
        The files below should be included in data_dir
        gtzan/genres_original/
        ├── blues
        ├── classical
        ├── ...
        └── rock
        '''
        self.data_dir = os.path.join(self.data_dir, 'genres_original')
        self.filepaths = []
        self.labels = []

        genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        for idx, genre in enumerate(genre_list):
            genre_dir = os.path.join(self.data_dir, genre)
            for fname in os.listdir(genre_dir):
                if fname.endswith('.wav'):
                    if fname == 'jazz.00054.wav':
                        continue # this file is corrupted
                    else:
                        self.filepaths.append(os.path.join(genre_dir, fname))
                        # idx is the label code for label name
                        self.labels.append(idx)

        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            self.filepaths, self.labels, test_size=0.2,
            stratify=self.labels, random_state=self.seed)
    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps, sample_rate, duration, n_fft, hop_length, n_mels, reduce_size):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps

                self.n_fft = n_fft
                self.hop_length = hop_length
                self.n_mels = n_mels

                self.sr = sample_rate
                self.duration = duration

                self.reduce_size = reduce_size

            def __getitem__(self, index):
                fpath = self.data[index]
                audio_info, sfr = librosa.load(fpath, sr=self.sr, duration=self.duration)
                
                # make the length equal
                max_len = int(self.sr * self.duration)
                if len(audio_info) < max_len:
                    # zero padding for not enough long audio file
                    audio_info = np.pad(audio_info, (0, max_len - len(audio_info)))
                else:
                    # slice for too long audio file
                    audio_info = audio_info[:max_len]

                # Mel-Spectrogram converting
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_info, 
                    sr=sfr, 
                    n_mels=self.n_mels, 
                    hop_length=self.hop_length, 
                    n_fft=self.n_fft,
                )
                inputs = librosa.power_to_db(mel_spec, ref=np.max)

                # resize to reduce dimension
                inputs = cv2.resize(inputs, (self.reduce_size, self.reduce_size), interpolation=cv2.INTER_LINEAR)
                inputs = torch.tensor(inputs, dtype=torch.float32).transpose(0, 1) # (C, W) - > (W, C)

                # coding (W, C) -> (T, W, C)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(
                self.train_data, self.train_targets, self.encode, self.T, self.sr, self.duration, self.n_fft, self.hop_length, self.n_mels, self.reduce_size)
        else:
            ds = DummyDataset(
                self.test_data, self.test_targets, self.encode, self.T, self.sr, self.duration, self.n_fft, self.hop_length, self.n_mels, self.reduce_size)

        return ds

    
@register_dataset('urbansound')
class iUrbanSound8K(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, reduce_size, seed)

        self.hop_length = 512
        self.n_mels = 128
        self.n_fft = 2048
        self.sr = 22050
        self.duration = 4.0

        self.num_classes = 10

    def download_data(self):
        # self.data_dir = /data/urbansound8k/
        '''
        there are 10-fold data in UrbanSound8K, we use the last two folder (9, 10) as test set in this repository
        urbansound8k/
        ├── audio/
        │   ├── fold1/
        │   ├── fold2/
        │   └── ...
        └── metadata/
            └── UrbanSound8K.csv
        '''
        csv_path = os.path.join(self.data_dir, 'metadata/UrbanSound8K.csv')
        audio_dir = os.path.join(self.data_dir, 'audio')

        annotations = pd.read_csv(csv_path)

        test_folders = [9, 10]
        train_annotations = annotations[~annotations['fold'].isin(test_folders)].reset_index(drop=True)
        test_annotations = annotations[annotations['fold'].isin(test_folders)].reset_index(drop=True)

        self.train_targets = train_annotations['classID'].values
        self.test_targets = test_annotations['classID'].values

        self.train_data = train_annotations.apply(lambda row: os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name']), axis=1).tolist()
        self.test_data = test_annotations.apply(lambda row: os.path.join(audio_dir, f"fold{row['fold']}", row['slice_file_name']), axis=1).tolist()
    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps, sample_rate, duration, n_fft, hop_length, n_mels, reduce_size):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps

                self.n_fft = n_fft
                self.hop_length = hop_length
                self.n_mels = n_mels

                self.sr = sample_rate
                self.duration = duration

                self.reduce_size = reduce_size

            def __getitem__(self, index):
                fpath = self.data[index]
                audio_info, sfr = librosa.load(fpath, sr=self.sr, duration=self.duration)
                
                # make the length equal
                max_len = int(self.sr * self.duration)
                if len(audio_info) < max_len:
                    # zero padding for not enough long audio file
                    audio_info = np.pad(audio_info, (0, max_len - len(audio_info)))
                else:
                    # slice for too long audio file
                    audio_info = audio_info[:max_len]

                # Mel-Spectrogram converting
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_info, 
                    sr=sfr, 
                    n_mels=self.n_mels, 
                    hop_length=self.hop_length, 
                    n_fft=self.n_fft,
                    fmin=20,
                    fmax=8000,
                    power=2.0
                )
                inputs = librosa.power_to_db(mel_spec, ref=np.max)

                # resize to reduce dimension
                inputs = cv2.resize(inputs, (self.reduce_size, self.reduce_size), interpolation=cv2.INTER_LINEAR)
                inputs = torch.tensor(inputs, dtype=torch.float32).transpose(0, 1) # (C, W) - > (W, C)

                # coding (W, C) -> (T, W, C)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(
                self.train_data, self.train_targets, self.encode, self.T, self.sr, self.duration, self.n_fft, self.hop_length, self.n_mels, self.reduce_size)
        else:
            ds = DummyDataset(
                self.test_data, self.test_targets, self.encode, self.T, self.sr, self.duration, self.n_fft, self.hop_length, self.n_mels, self.reduce_size)

        return ds

    
@register_dataset('gsc')
class iGoogleSpeechCommands(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, reduce_size, seed)

        self.hop_length = 160
        self.n_mels = 128
        self.n_fft = 400
        self.sr = 16000
        self.duration = 1.0

        self.num_classes = 35

    def download_data(self):
        '''
        After download and unzip automatically by Torch, the file directory structure is
        /SpeechCommands/speech_commands_v0.02/
        ├── backward
        ├── bed
        ├── bird
        ├── ...
        ├── yes
        └── zero
        '''
        # self.data_dir = data/gsc/
        # V2 data as default
        CORE_WORDS = ["yes","no","up","down","left","right","on","off","stop","go"]
        AUX_WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine",
                    "bed","bird","cat","dog","happy","house","marvin","sheila","tree",
                    "wow","backward","forward","follow","learn","visual"]
        ALL_WORDS = CORE_WORDS + AUX_WORDS
        assert len(ALL_WORDS) == 35
        # BACKGROUND_DIR = "_background_noise_"

        paths = []
        labels = []
        self.label_map = {}

        base_dir = os.path.join(self.data_dir, 'SpeechCommands', 'speech_commands_v0.02')

        self.label_map = {w: i for i, w in enumerate(sorted(ALL_WORDS))}
        # self.label_map["_silence_"] = len(labels) # background_noise -> silence

        for lbl in ALL_WORDS:
            dirpath = os.path.join(base_dir, lbl)
            if not os.path.isdir(dirpath): continue
            for fname in os.listdir(dirpath):
                if fname.endswith('.wav'):
                    paths.append(os.path.join(dirpath, fname))
                    labels.append(self.label_map[lbl])


        self.train_data, self.test_data, self.train_targets, self.test_targets = train_test_split(
            paths, labels, 
            test_size=0.2,
            stratify=labels, 
            random_state=self.seed
        )

    
    def get_dataset(self, train=True):
        class DummyDataset(Dataset):
            def __init__(self, data, targets, encode, time_steps, sample_rate, duration, n_fft, hop_length, n_mels, reduce_size):
                self.data = data
                self.targets = targets

                self.encode = encode
                self.time_steps = time_steps

                self.n_fft = n_fft
                self.hop_length = hop_length
                self.n_mels = n_mels

                self.sr = sample_rate
                self.duration = duration

                self.reduce_size = reduce_size

            def __getitem__(self, index):
                fpath = self.data[index]
                audio_info, sfr = librosa.load(fpath, sr=self.sr, duration=self.duration)
                
                # make the length equal
                max_len = int(self.sr * self.duration)
                if len(audio_info) < max_len:
                    # zero padding for not enough long audio file
                    audio_info = np.pad(audio_info, (0, max_len - len(audio_info)))
                else:
                    # slice for too long audio file
                    audio_info = audio_info[:max_len]

                # Mel-Spectrogram converting
                mel_spec = librosa.feature.melspectrogram(
                    y=audio_info, 
                    sr=sfr, 
                    n_mels=self.n_mels, 
                    hop_length=self.hop_length, 
                    n_fft=self.n_fft,
                    fmin=20,
                    fmax=self.sr // 2,
                    power=2.0
                )
                inputs = librosa.power_to_db(mel_spec, ref=np.max)

                # resize to reduce dimension
                inputs = cv2.resize(inputs, (self.reduce_size, self.reduce_size), interpolation=cv2.INTER_LINEAR)
                inputs = torch.tensor(inputs, dtype=torch.float32).transpose(0, 1) # (C, W) - > (W, C)

                # coding (W, C) -> (T, W, C)
                x = coding_map[self.encode](inputs, num_steps=self.time_steps)
                y = self.targets[index]

                return x, y

            def __len__(self):
                return len(self.targets)

        if train:
            ds = DummyDataset(
                self.train_data, self.train_targets, self.encode, self.T, self.sr, self.duration, self.n_fft, self.hop_length, self.n_mels, self.reduce_size)
        else:
            ds = DummyDataset(
                self.test_data, self.test_targets, self.encode, self.T, self.sr, self.duration, self.n_fft, self.hop_length, self.n_mels, self.reduce_size)

        return ds
    

@register_dataset('shd')
class iSpikingHeidelbergDigits(AudioData):
    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, reduce_size, seed)

        self.duration = 150 # window size
        self.n_mels = 700 # fixed channel number

        self.num_classes = 20

    def _preprocess(self, times, units, label):
        data_label = torch.tensor(label, dtype=torch.int64)
        max_unit   = self.n_mels
        max_time   = 1
        dt         = 1 / self.duration 
        time_frames  = int(max_time / dt) # the total samping frames
        list_input = []
        for i in range(time_frames):
            indexs = np.argwhere(times <= i * dt).flatten()
            vals   = units[indexs]; vals = vals[vals > 0]
            vector = np.zeros(max_unit); vector[max_unit - vals] = 1
            times  = np.delete(times, indexs)
            units  = np.delete(units, indexs)
            list_input.append(vector)

        # resize to reduce dimension
        inputs = cv2.resize(np.array(list_input), (self.reduce_size, self.reduce_size), interpolation=cv2.INTER_LINEAR)
        
        # to tensor
        data_input = torch.tensor(inputs, dtype=torch.float32)

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
                # (W, C)
                inputs = self.data[index]

                # coding (W, C) -> (T, W, C)
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
    def __init__(self, data_dir, coding_schema, time_step, reduce_size, seed=2025):
        super().__init__(data_dir, coding_schema, time_step, reduce_size, seed)

        self.duration = 150
        self.n_mels = 700 # channel number

        self.num_classes = 35

    def _preprocess(self, times, units, label):
        data_label = torch.tensor(label, dtype=torch.int64)
        max_unit   = self.n_mels # this is the channel number
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

        # resize to reduce dimension
        inputs = cv2.resize(np.array(list_input), (self.reduce_size, self.reduce_size), interpolation=cv2.INTER_LINEAR)

        data_input = torch.tensor(inputs, dtype=torch.float32)
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
                # (W, C)
                inputs = self.data[index]

                # coding (W, C) -> (T, W, C)
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
