Lightweight
======

In this chapter, we will introduce the lightweighting-related APPs provided by the Soul toolkit.

Structured Pruning
------------------------------
Introduction:
    The method of this APP is derived from the paper 《Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning》.
    It proposes a novel pruning framework that combines unstructured weight pruning and unstructured neuron pruning to maximize the utilization of 
    sparsity in neuromorphic computing, thereby improving energy efficiency.

Citation:
    Yaxin Li et al., Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning

Usage:

.. code-block:: bash

    python structured_prune.py 
    --mode load 
    --workers 4 
    --seed 2025 
    --log_dir ./logs 
    --data_dir ./data/cifar10 
    --model_dir ./saved_models 
    --epochs 150 
    --batch_size 128 
    --optimizer adam 
    --scheduler cosine 
    --learning_rate 0.0001 
    --weight_decay 0.0 
    --momentum 0.9 
    --model spikingvgg9 
    --neuron_type lif 
    --time_step 4 
    --dataset_name cifar10 
    --model_path ./saved_models/best_spikingvgg9_lif_cifar10_2099.pt 
    --pruning_nodes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

Parameter Description:

Most parameters of ``structured_prune.py`` are consistent with the general process of training models with Soul. For questions, please refer to :doc:`../params`
Only the special parameters of ``structured_prune.py`` are introduced here:

* mode： ``structured_prune.py`` supports two modes:
    * train: Train the model directly and perform pruning after the model training is completed. 
    * load: Load existing model parameters and perform pruning on the model.
* model_path：Specify where to read model parameters when the ``--mode`` parameter is 'load'.
* pruning_nodes：Indices of neuron nodes to be pruned, where only neuron nodes are considered for the indices.

Unstructured Pruning
-----------------------------------------
Introduction:
    The method of this APP is derived from the paper《Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework》， 
    It proposes a structured pruning method based on the activity level of convolutional kernels,referred to as the Spike Channel Activity (SCA)-based 
    network pruning framework. Inspired by the synaptic plasticity mechanism, our method dynamically adjusts the network structure by pruning and 
    regenerating convolutional kernels during training, thereby enhancing the model's adaptability to the current target task.While maintaining model 
    performance, this method optimizes the network architecture, ultimately reducing computational load and accelerating the inference process.

Citation:
    Zhaofei Yu et al., Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework

Usage:

.. code-block:: bash

    python unstructured_prune.py 
    --workers 4 
    --seed 2025 
    --log_dir ./logs 
    --data_dir ./data/cifar10 
    --model_dir ./saved_models 
    --epochs 150 
    --batch_size 128 
    --optimizer adam 
    --scheduler cosine 
    --learning_rate 0.0001 
    --weight_decay 0.0 
    --momentum 0.9 
    --model spikingvgg9 
    --neuron_type lif 
    --time_step 4 
    --dataset_name cifar10 
    --neuron_prune_layers [1,2,3,4,5] 
    --weight_prune_layers [0,1,2,3,4,5,6,7] 
    --mask_init_factor [0, 0, 0, 0] 
    --criterion MSE

Parameter Description:

Most parameters of ``unstructured_prune.py`` are consistent with the general process of training models with Soul. For questions, please refer to :doc:`../params`
Only the special parameters of ``unstructured_prune.py`` are introduced here:

* TET：Whether to use the TET factor in the loss function.
* TET_phi：The phi value of the TET factor.
* TET_lambda：The lambda value of the TET factor.
* save_latest：Whether to save the final pruning results (the entire model).
* not_prune_weight：Whether to skip pruning for weights.
* not_prune_neuron：Whether to skip pruning for neurons.
* finetune：Whether to enable the fine-tuning process.
* prune_optimizer：The optimizer used in the pruning process; options: adam, sgd.
* prune_lr： The initial learning rate used in the pruning process.
* prune_weight_decay：The weight decay rate used in the pruning process.
* penalty_lmbda：The lambda value of the penalty term used in the loss calculation.
* accumulate_step：The training optimization step size.
* temp_scheduler：  Initialization parameters for the temperature scheduler, used in the following formats:

    --temp_scheduler <init temp> <final temp>
    or --temp_scheduler <init temp> <final temp> <T0> <Tmax>
    or --temp_scheduler <init temp of weight> <init temp of neuron> <final temp of weight> <final temp of neuron> <T0> <Tmax>

* finetune_lr：The initial learning rate used in the fine-tuning process.
* epoch_finetune：The number of epochs for the fine-tuning process.
* finetune_lr_scheduler：Scheduler settings for fine-tuning, used in the following formats:

    --scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
    or --scheduler Step [minestones]

* neuron_prune_layers：Indices of neuron nodes to be considered for pruning.
* weight_prune_layers：Indices of convolutional layers to be considered for pruning.
* mask_init_factor：Initialization settings for the pruning mask, used in the following format:

    --mask-init-factor <weights mean> <neurons mean> <weights std> <neurons std>

* criterion：The criterion for loss calculation; options: MSE, CE.
