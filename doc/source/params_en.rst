Parameter Settings
========

In this chapter, we will learn how to set parameters for the training process.
In the Soul toolkit, there are three ways to set parameters for the training process:

* Pass parameters corresponding to the ``soul.utils.parser.parse_args()`` method
* Modify the yaml files in ``soul.config``
* Directly modify the training source code

soul.utils.parser.parse_args Method
------------------------------------------------

* --workers，-j：Number of subprocesses for loading data
* --seed，-seed：Random seed
* --log_dir，-log_dir：Log output path
* --data_dir，-data_dir：Dataset storage path
* --model_dir，-model_dir：Model saving path
* --epochs，-epochs：Number of training epochs
* --batch_size，-b：Training batch size
* --learning_rate，-lr：Learning rate
* --weight_decay，-wd：Weight decay magnitude
* --momentum，-momentum：Momentum coefficient of the optimizer
* --optimizer，-optimizer：Optimizer to be used, with the following options:

    * sgd：SGD optimizer
    * adam：Adam optimizer
    * adamw：AdamW optimizer
    * rmsprop：RMSprop optimizer

* --scheduler，-scheduler：Training scheduler to be used, with the following options:

    * cosine：CosineAnnealingLR scheduler
    * linear：StepLR scheduler
    * warmup：CosineAnnealingWarmRestarts scheduler

* --dataset_name，-dataset：Dataset used for training. Currently, Soul supports the following datasets:

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

For more detailed introductions to different datasets of each modality, please refer to :doc:`../datasets/index`

* --model，-m：Type of spiking neural network model to be trained. Currently, Soul supports the following models:

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

* --coding_schema，-coding：Coding scheme adopted for static raw input, with the following optional coding methods:

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

For a more detailed introduction to encoding, please refer to :doc:`../encoding`

* --neuron_type，-n：Type of neuron adopted. Currently, Soul supports the following neurons:

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

For a more detailed introduction to neurons, please refer to :doc:`../neuron`

*  --time_step，-T：Time step for inference of the spiking neural network model
*  --noise_type，-noise：Noise that can be set during experimental analysis (not applicable to neuromorphic sensing tasks), with optional values:

    * gaussian
    * dropouts

* --noise_intensity，-ni：Noise intensity (sigma value of Gaussian noise, dropout rate of the dropout layer)
* --noise_phase，-phase：Mode for adding noise

    * train
    * test

soul.config.basic.yaml Basic Configuration
------------------------------------------------

* workers：Number of worker processes for processing data
* seed：Random seed
* state：Log level
* log_dir：Output log path
* model_dir：Model saving path
* data_dir： Input dataset path
* sop：Whether to simulate the energy consumption of neuromorphic chips
* e_ac：Energy consumption cost/picojoules (pJ) for running accumulation operations on hypothetical 45nm hardware
* e_mac：Energy consumption cost/picojoules (pJ) for running multiply-accumulate operations on hypothetical 45nm hardware
* application：Default application scenario (modality type), with optional values:

    * motion
    * vision
    * acoustic
    * wireless

* epochs：Number of training epochs
* batch_size：Training batch size
* optimizer：Optimizer used for training, with the following options:

    * sgd：SGD optimizer
    * adam：Adam optimizer
    * adamw：AdamW optimizer
    * rmsprop：RMSprop optimizer

* scheduler：Scheduler used for training, with the following options:

    * cosine：CosineAnnealingLR scheduler
    * linear：StepLR scheduler
    * warmup：CosineAnnealingWarmRestarts scheduler

* learning_rate：learning rate
* weight_decay：Weight decay value of the optimizer (L2 regularization penalty term)
* momentum：Momentum coefficient of the optimizer
* model：Model used for training. Currently, Soul supports the following models:

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

* neuron_type：Type of neuron used. Currently, Soul supports the following neurons:

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

* time_step：Time step of the spiking neural network
* dataset_name：Type of dataset used. Currently, Soul supports the following datasets:

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

* coding_schema：Coding scheme for static raw input, with the following optional coding methods:

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

* receptor_size：Size of DVS images
* window_size：Size of sliding window sampling for motion sensing datasets
* step_size： Step size of sliding window sampling for motion sensing datasets
* reduce_size：Reduction size of original mel spectrograms in acoustic sensing

soul.config.model.application.*.yaml Model Configuration
------------------------------------------------
The content of the configuration files under ``soul.config.model`` guides Soul to build the specified model. If you want to modify some model configurations, you can refer to the corresponding model in the ``soul.model`` package to modify the parameters in the configuration file during the initialization process.
For an in-depth understanding of the parameter meanings of a specific model, please refer to :doc:`../models/index`

soul.config.neuron.*.yaml Neuron Configuration
------------------------------------------------
The content of the configuration files under ``soul.config.neuron`` guides Soul to build the specified neuron. If you want to modify some neuron configurations, you can refer to the corresponding neuron initialization process in the ``soul.neuron`` package to modify the parameters in the configuration file.
For an in-depth understanding of the parameter meanings of a specific neuron, please refer to :doc:`../neuron`