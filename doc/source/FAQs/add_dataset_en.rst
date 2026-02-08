How to Add a Dataset
============================

This tutorial will show you how to add a local dataset to the Soul project.
First, we need to understand how the Soul toolkit creates a dataset based on our settings:

.. code-block:: python

    train_dataset, test_dataset = load_dataset(config)

The Soul toolkit directly reads the ``config`` configuration information we set to initialize the dataset. For details on how the ``config`` configuration information is read, please refer to :doc:`../params`.
We can check the operations it performs in ``soul.utils.dataset.__init__.py``.

.. code-block:: python

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
    from .vision_dataset import iCIFAR10, iCIFAR100, iCIFAR10DVS, iDVSGesture, iTinyImageNet, iFashionMNIST, iSVHN, iMNIST, iNMNIST, iNCaltech101
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

As can be seen, the ``load_dataset(config)`` method first initializes a dataset instance through the ``get_idata(config)`` method. This method reads the corresponding dataset class from ``_REGISTRY_DATASET``(e.g., the CIFAR100 dataset is registered in ``_REGISTRY_DATASET`` via the  ``@register_dataset('cifar100')`` decorator)，and calls the initialization method of this class to obtain the dataset instance.
Then it calls the ``dataset.download_data()`` method to load local dataset data (not all datasets provide an online download function). Finally, depending on the modality the dataset belongs to, the ``load_dataset(config)`` method sets variables in the ``config`` object that will be used during inference, and generates the training and test datasets.

So far, we know that a dataset class in the Soul toolkit needs to expose three main methods:
    
    * 1. The ``dataset.__init__()`` initialization method, which initializes some necessary variables
    * 2. The ``dataset.download_data()`` method for loading local data, which reads the raw data and labels of the training and test data
    * 3. The ``dataset.get_dataset(train)`` method for obtaining the training or test dataset, which preprocesses the raw data and wraps it as a ``torch.utils.data.dataset.Dataset`` class to return for use in the inference process 

Meanwhile, depending on the modality, different variables need to be defined for reference in subsequent inference processes:

    * Image recognition: ``input_channels`` (number of input channels), ``input_height``  (input height), ``input_width`` (input width),  ``num_classes`` (number of classes)
    * Wireless signal processing:  ``input_channels`` (number of input channels), ``input_height`` (input height), ``input_width`` (input width),  ``num_classes`` (number of classes)
    * Sound recognition: ``input_dim`` (sampling window size),  ``input_channels``  (number of feature dimensions), ``num_classes`` (number of classes)
    * Motion recognition: ``input_dim`` (reduced size of the original Mel spectrogram),  ``input_channels``  (reduced size of the original Mel spectrogram, consistent with ``input_dim`` as  ``reduce_size`` ), ``num_classes`` (number of classes)

Of course, we also need to add the decorator ``@register_dataset('my_dataset')`` to the custom dataset class to register the dataset ('my_dataset' is the custom dataset name), and import the custom dataset class in ``soul.utils.dataset.__init__.py``.