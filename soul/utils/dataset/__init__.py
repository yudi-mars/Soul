__all__ = ['load_dataset']

_REGISTRY_DATASET = {}
def register_dataset(name):
    def dec(cls):
        _REGISTRY_DATASET[name.lower()] = cls
        return cls
    return dec

def get_idata(config): 
    name = config['dataset_name'].lower()
    cls = _REGISTRY_DATASET.get(name)

    if cls is None:
        raise NotImplementedError(f"Unknown dataset {name!r}.")
    import inspect
    sig = inspect.signature(cls.__init__)
    kwargs = {k: config[k] for k in config if k in sig.parameters and k != 'self'}
    return cls(**kwargs)

# this is important to hook registery
from .motion_dataset import iHHAR, iMotionSense, iShoaib, iUCIHAR
from .vision_dataset import iCIFAR10, iCIFAR100, iCIFAR10DVS, iDVSGesture, iTinyImageNet, iFashionMNIST, iSVHN, iMNIST
from .acoustic_dataset import iSpikingHeidelbergDigits, iSpikingSpeechCommands, iUrbanSound8K, iGoogleSpeechCommands, iGTZAN, iESC50
from .wireless_dataset import iFiHAR, iFiHumanID, iUTHAR, iWidar3, iWiGesture, iFallDar, iARIL, iBullyDetect

def load_dataset(config):
    idata = get_idata(config)
    idata.download_data()

    # update input shape of different applications for config
    assert idata.input_shape is not None, 'Something wrong'
    if config['application'] == 'vision': # -> frame (C, H, W)
        config['input_channels'], config['input_height'], config['input_width'] = idata.input_shape
    elif config['application'] == 'wireless': # -> CSI (C, H, W)
        config['input_channels'], config['input_height'], config['input_width'] = idata.input_shape
    elif config['application'] == 'motion': # -> multi-series (C, W)
        config['input_channels'], config['input_dim'] = idata.input_shape
    elif config['application'] == 'acoustic': # -> Mel-spectrogram (W, C)
        config['input_dim'], config['input_channels'] = idata.input_shape
    else:
        raise ValueError(f'Unknown application type: {config["application"]}')
    
    # update num_classes for config
    config['num_classes'] = idata.num_classes

    # load dataset 
    train_dataset = idata.get_dataset(train=True)
    test_dataset = idata.get_dataset(train=False)

    return train_dataset, test_dataset
