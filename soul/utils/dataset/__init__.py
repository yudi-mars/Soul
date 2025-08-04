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
from .vision_dataset import iCIFAR10, iCIFAR100, iCIFAR10DVS, iDVSGesture, iTinyImageNet
from .acoustic_dataset import iSpikingHeidelbergDigits, iSpikingSpeechCommands, iUrbanSound8K, iGoogleSpeechCommands, iGTZAN

def load_dataset(config):
    idata = get_idata(config)
    idata.download_data()

    # update input shape of different applications for config
    assert idata.input_shape is not None, 'Something wrong'
    if config['application'] == 'vision':
        config['input_channels'], config['input_height'], config['input_width'] = idata.input_shape
    elif config['application'] == 'motion':
        config['input_channels'], config['input_dim'] = idata.input_shape
    elif config['application'] == 'acoustic':
        config['input_channels'], config['input_dim'] = idata.input_shape
    else:
        raise ValueError(f'Unknown application type: {config["application"]}')
    
    # update num_classes for config
    config['num_classes'] = idata.num_classes

    # load dataset 
    train_dataset = idata.get_dataset(train=True)
    test_dataset = idata.get_dataset(train=False)

    return train_dataset, test_dataset
