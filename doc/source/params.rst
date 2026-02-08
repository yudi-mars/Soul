参数设置
========

本章内容中，我们将了解如何为训练流程设置参数。
Soul工具包中，为训练流程设置参数的方式分为三种：

* 根据 ``soul.utils.parser.parse_args()`` 方法中的对应参数传参
* 修改 ``soul.config`` 中的yaml文件
* 直接修改训练源码

soul.utils.parser.parse_args方法
------------------------------------------------

* --workers，-j：加载数据的子进程数
* --seed，-seed：随机种子
* --log_dir，-log_dir：日志输出地址
* --data_dir，-data_dir：数据集所在地址
* --model_dir，-model_dir：模型保存地址
* --epochs，-epochs：训练次数
* --batch_size，-b：训练批次大小
* --learning_rate，-lr：学习率
* --weight_decay，-wd：权重衰减幅度
* --momentum，-momentum：优化器的惯性系数
* --optimizer，-optimizer：使用的优化器，可选项如下

    * sgd：SGD优化器
    * adam：Adam优化器
    * adamw：AdamW优化器
    * rmsprop：RMSprop优化器

* --scheduler，-scheduler：使用的训练调度器，可选项如下

    * cosine：CosineAnnealingLR调度器
    * linear：StepLR调度器
    * warmup：CosineAnnealingWarmRestarts调度器

* --dataset_name，-dataset：训练使用的数据集，目前Soul支持的数据集如下：

    * motion sensing：
        CIFAR-10、CIFAR-100、SVHN、Tiny-ImageNet、MNIST、Fashion-MNIST
    * vision sensing：
        UCI HAR、HHAR、MotionSense、Shoaib
    * acoustic sensing：
        UrbanSound8K、GSC、GTZAN、ESC-50
    * wireless sensing：
        UT-HAR、NTU-HumanID、BullyDetect、ARIL、NTU-HAR、Widar 3.0
    * Neuromorphic sensing：
        CIFAR10-DVS、DVS-Gesture、SHD、SSC

关于各个模态的不同数据集更加详细的介绍，请参考 :doc:`../datasets/index`

* --model，-m：将训练的脉冲神经网络模型类型，目前Soul支持的模型如下：

    * motion sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、DCNN、SenseHAR、DCL、
        ISpikformer、ISpikformer256、ISpikformer384、ISpikformer512、SpikingVGG9、
        SpikingVGG16、SpikingVGG19、SEWResNet34、SEWResNet50、MSResNet34、
        MSResNet50、Spikformer256、Spikformer384、MetaSpikeformer256、MetaSpikeformer384、
        QKFormer256、QKFormer512
    * vision sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG5、SpikingVGG9、
        SpikingVGG11、SpikingVGG13、SpikingVGG16、SpikingVGG19、SEWResNet18、SEWResNet50、
        SEWResNet34、MSResNet18、MSResNet34、MSResNet50、Spikformer256、Spikformer384、
        Spikformer512、MetaSpikeformer256、MetaSpikeformer384、MetaSpikeformer512、
        SpikingResformer192、SpikingResformer256、SpikingResformer384、SpikingResformer512、
        QKFormer256、QKFormer384、QKFormer512
    * acoustic sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG9、SpikingVGG16、
        SEWResNet18、SEWResNet50、MSResNet18、MSResNet50、SpikingTCN、Spikformer256、
        Spikformer384、Spikformer512、MetaSpikeformer256、MetaSpikeformer384、MetaSpikeformer512、
        SpikingResformer192、SpikingResformer256、SpikingResformer384、SpikingResformer512、
        QKFormer256、QKFormer384、QKFormer512
    * wireless sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingTCN、SpikingVGG9、
        SpikingVGG16、SEWResNet34、SEWResNet50、MSResNet34、MSResNet50、Spikformer256、
        Spikformer384、MetaSpikeformer256、MetaSpikeformer384、QKFormer256、QKFormer384、
        SpikingResformer256、SpikingResformer384

* --coding_schema，-coding：采用的静态原始输入的编码方案，可选的编码方式如下：

    * binary_par_encode
    * binary_seq_encode
    * bsa_encode
    * burst_encode
    * direct_code
    * phase_encode
    * poisson_encode
    * population_encode
    * rank_order_encode
    * temporal_encode
    * rate_encode
    * sdr_encode
    * tcr_mwc_encode
    * tcr_sf_decode
    * tcr_tbr_encode

关于神经元的更加详细的介绍，请参考 :doc:`../encoding`

* --neuron_type，-n：采用的神经元类型，目前Soul支持的模型如下：

    * LIFNode
    * ParametricLIFNode
    * CLIFNode
    * GatedLIFNode
    * INTLIFNode
    * ParallelSpikingNode
    * TLIFNode
    * IELIFNode
    * LTMD
    * STBIF
    * ILIFNeuron
    * RPLIFNode

关于神经元的更加详细的介绍，请参考 :doc:`../neuron`

*  --time_step，-T：脉冲神经网络模型推理时的时间步
*  --noise_type，-noise：在实验分析时可设置的噪声（不适用于neuromorphic sensing任务），可选项有：

    * gaussian
    * dropouts

* --noise_intensity，-ni：噪声强度（高斯噪声的sigma值、dropout层的丢弃率）
* --noise_phase，-phase：添加噪声的模式

    * train
    * test

soul.config.basic.yaml基础配置
------------------------------------------------

* workers：处理数据的工作进程数量
* seed：随机种子
* state：日志等级
* log_dir：输出日志地址
* model_dir：保存模型的地址
* data_dir：输入数据集的地址
* sop：是否模拟神经形态芯片的能耗
* e_ac：在假定的 45 纳米硬件上，累加操作运行的能耗成本/皮焦（pJ）
* e_mac：在假定的 45 纳米硬件上，乘法累加操作的能耗成本/皮焦（pJ）
* application：默认应用场景（模态类型），可选项有：

    * motion
    * vision
    * acoustic
    * wireless

* epochs：训练轮次数量
* batch_size：训练批次大小
* optimizer：训练所用优化器，可选项如下

    * sgd：SGD优化器
    * adam：Adam优化器
    * adamw：AdamW优化器
    * rmsprop：RMSprop优化器

* scheduler：训练所用调度器，可选项如下

    * cosine：CosineAnnealingLR调度器
    * linear：StepLR调度器
    * warmup：CosineAnnealingWarmRestarts调度器

* learning_rate：学习率
* weight_decay：优化器的权重衰减值（L2 正则化惩罚项）
* momentum：优化器的惯性系数
* model：训练所用模型，目前Soul支持的模型如下：

    * motion sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG5、
        SpikingVGG9、SpikingVGG11、SpikingVGG13、SpikingVGG16、SpikingVGG19、
        SEWResNet18、SEWResNet34、SEWResNet50、MSResNet18、MSResNet34、
        MSResNet50、Spikformer256、Spikformer384、Spikformer512、MetaSpikeformer256、
        MetaSpikeformer384、MetaSpikeformer512、SpikingResformer192、SpikingResformer256、
        SpikingResformer384、SpikingResformer512、QKFormer256、QKFormer384、QKFormer512
    * vision sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingVGG9、
        SpikingVGG16、SEWResNet18、SEWResNet50、MSResNet18、MSResNet50、
        SpikingTCN
    * acoustic sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、DCNN、
        SenseHAR、DCL、ISpikformer
    * wireless sensing：
        SpikingMLP、SpikingLeNet、SpikingRNN、SpikingConvRNN、SpikingTCN

* neuron_type：所用神经元类型，目前Soul支持的模型如下：

    * LIFNode
    * ParametricLIFNode
    * CLIFNode
    * GatedLIFNode
    * INTLIFNode
    * ParallelSpikingNode
    * TLIFNode
    * IELIFNode
    * LTMD
    * STBIF
    * ILIFNeuron
    * RPLIFNode

* time_step：脉冲神经网络的时间步长
* dataset_name：所用数据集类型，目前Soul支持的数据集如下：

    * motion sensing：
        CIFAR-10、CIFAR-100、SVHN、Tiny-ImageNet、MNIST、Fashion-MNIST
    * vision sensing：
        UCI HAR、HHAR、MotionSense、Shoaib
    * acoustic sensing：
        UrbanSound8K、GSC、GTZAN、ESC-50
    * wireless sensing：
        UT-HAR、NTU-HumanID、BullyDetect、ARIL、NTU-HAR、Widar 3.0
    * Neuromorphic sensing：
        CIFAR10-DVS、DVS-Gesture、SHD、SSC

* coding_schema：静态原始输入的编码方案，可选的编码方式如下：

    * binary_par_encode：
    * binary_seq_encode：
    * bsa_encode：
    * burst_encode：
    * direct_code：
    * phase_encode：
    * poisson_encode：
    * population_encode：
    * rank_order_encode：
    * temporal_encode：
    * rate_encode：
    * sdr_encode：
    * tcr_mwc_encode：
    * tcr_sf_decode：
    * tcr_tbr_encode：

* receptor_size：DVS图像的大小
* window_size：运动传感数据集的滑动窗口采样的大小
* step_size：运动传感数据集的滑动窗口采样的补偿
* reduce_size：音频传感中原始梅尔频谱图的缩减尺寸

soul.config.model.application.*.yaml模型配置
------------------------------------------------
``soul.config.model`` 下的配置文件的内容指导Soul构建指定的模型，若您想要修改某些模型配置，可以参考 ``soul.model`` 包中对应的模型修改初始化过程配置文件中的参数
若想深入了解某个模型的参数意义，请参考 :doc:`../models/index`

soul.config.neuron.*.yaml神经元配置
------------------------------------------------
``soul.config.neuron`` 下的配置文件的内容指导Soul构建指定的神经元，若您想要修改某些神经元配置，可以参考 ``soul.neuron`` 包中对应的神经元初始化过程修改配置文件中的参数
若想深入了解某个神经元的参数意义，请参考 :doc:`../neuron`