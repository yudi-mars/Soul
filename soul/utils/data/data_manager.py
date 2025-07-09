import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from .vision_dataset import iCIFAR10, iCIFAR100, iTinyImageNet, iCIFAR10DVS, iDVSGesture
from soul.utils.coding import coding_map

class DummyDataset(Dataset):
    def __init__(self, inputs, labels, trsf, source='vision-npy', config=None):
        super().__init__()

        self.inputs = inputs
        self.labels = labels
        self.trsf = trsf

        self.source = source
        self.encode = config['coding_schema']
        self.T = config['time_step']

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):

        x = self.trsf(self._file_loader(self.inputs[index], self.source))
        if 'dvs' not in self.source: # TODO maybe other sensing also donot need encoding
            x = coding_map[self.encode](x, num_steps=self.T)
        y = self.labels[index]

        return x, y
    
    def _file_loader(self, path, source):
        if source == 'vision-rgb':
            # read from picture file
            img = Image.open(open(path, "rb")) 
            return img.convert("RGB") # PIL.image
        elif source == 'vision-npy':
            return Image.fromarray(path) # PIL.image from array, shape (C, H, W)
        elif source == 'vision-dvs':
            return torch.from_numpy(np.load(path)['frames']).float() # directly to Tensor with shape (T, C, H, W)
        else:
            return path

class DataManager(object):
    def __init__(self, config):
        self.input_shape = None
        self.config = config

        self._setup_data()

    def _setup_data(self):
        idata = _get_idata(self.config['dataset_name'], self.config['data_dir'], self.config['time_step'])
        idata.download_data()

        # data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # shape
        self.input_shape = idata.input_shape
        self.num_classes = idata.num_classes

        # input data source
        self.data_source = f"{self.config['application']}-{idata.data_source}"

    def update_config(self):
        assert self.input_shape is not None, 'Something wrong'
        if self.config['application'] == 'vision':
            self.config['input_channels'], self.config['input_height'], self.config['input_width'] = self.input_shape
        elif self.config['application'] == 'motion':
            # TODO
            pass
        elif self.config['application'] == 'acoustic':
            # TODO
            pass
        else:
            raise ValueError(f'Unknown application type: {self.config["application"]}')
        
        self.config['num_classes'] = self.num_classes
        
        return self.config 

    def get_dataset(self):
        train_x, train_y = self._train_data, self._train_targets
        test_x, test_y = self._test_data, self._test_targets

        train_trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        test_trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])

        train_dataset = DummyDataset(train_x, train_y, train_trsf, self.data_source, self.config)
        test_dataset = DummyDataset(test_x, test_y, test_trsf, self.data_source, self.config)

        return train_dataset, test_dataset

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

def _get_idata(dataset_name, dataset_dir, T):
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10(dataset_dir, T)
    elif name == 'cifar100':
        return iCIFAR100(dataset_dir, T)
    elif name == 'imagenet':
        return iTinyImageNet(dataset_dir, T)
    elif name == 'cifar10dvs':
        return iCIFAR10DVS(dataset_dir, T)
    elif name == 'dvsgesture':
        return iDVSGesture(dataset_dir, T)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))