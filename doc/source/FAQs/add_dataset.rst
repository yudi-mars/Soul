如何添加数据集
============================

本教程将告诉您如何添加一个本地的数据集到Soul项目中。
首先，我们需要了解Soul工具包是如何通过我们的设置创建一个数据集的：

.. code-block:: python

    train_dataset, test_dataset = load_dataset(config)

Soul工具包直接读取我们设置的 ``config`` 配置信息来初始化数据集，关于 ``config`` 配置信息如何读取，详情见 :doc:`../params`。
我们可以在 ``soul.utils.dataset.__init__.py`` 中查看它做了哪些操作。

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

可以看到 ``load_dataset(config)`` 方法首先通过 ``get_idata(config)`` 方法初始化数据集实例，该方法从 ``_REGISTRY_DATASET`` 中读取对应数据集类（如CIFAR100数据集通过 ``@register_dataset('cifar100')`` 注释注册到 ``_REGISTRY_DATASET`` 中），并调用该类的初始化方法获得数据集实例。 
之后调用 ``dataset.download_data()`` 方法加载本地数据集数据（并非所有数据集都提供在线下载功能）。最后，根据数据集所属模态的不同， ``load_dataset(config)`` 方法在 ``config`` 对象中设置推理过程中会使用到的变量并生成训练集和测试集。

目前为止，我们知道Soul工具包中的数据集类需要暴露三个主要方法：
    
    * 1. ``dataset.__init__()`` 初始化方法，初始化一些必要变量
    * 2. ``dataset.download_data()`` 加载本地数据的方法，读取训练数据与测试数据的原始数据和标签
    * 3. ``dataset.get_dataset(train)`` 获取训练集或数据集的方法，预处理原始数据并包装为 ``torch.utils.data.dataset.Dataset`` 类返回供推理流程使用 

同时，根据模态的不同，需要定义不同的变量供之后的推理过程参考:

    * 图像识别： ``input_channels`` 输入通道数、 ``input_height`` 输入高度、 ``input_width`` 输入宽度、 ``num_classes`` 类别数量
    * 无线信号处理：： ``input_channels`` 输入通道数、 ``input_height`` 输入高度、 ``input_width`` 输入宽度、 ``num_classes`` 类别数量
    * 声音识别： ``input_dim`` 采样窗口大小、 ``input_channels`` 特征维度数、 ``num_classes`` 类别数量
    * 动作识别： ``input_dim`` 原始梅尔频谱图的缩小尺寸、 ``input_channels`` 原始梅尔频谱图的缩小尺寸（与 ``input_dim`` 一致为 ``reduce_size`` ）、 ``num_classes`` 类别数量

当然，我们还需要在自定义数据集类上添加注释 ``@register_dataset('my_dataset')`` 注册数据集（'my_dataset'为自定义的数据集名称），并在 ``soul.utils.dataset.__init__.py`` 中import自定义的数据集类