轻量化
======

本章，我们将介绍Soul工具包提供的轻量化相关的APP。

结构化剪枝（structured_prune）
------------------------------
介绍：
    该APP的方法来源于论文《Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning》，
    它提出了一种新的剪枝框架，该框架结合了非结构化权重剪枝和非结构化神经元剪枝，以最大限度地利用神经形态计算的稀疏性，从而提高能效。

引用：
    Yaxin Li et al., Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning

使用方法：

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

参数说明：

``structured_prune.py`` 的大部分参数与Soul训练模型的一般流程相似，如有疑问请参考 :doc:`../params`
此处仅介绍 ``structured_prune.py`` 的特殊参数：

* mode： ``structured_prune.py`` 支持两种模式：
    * train：直接训练模型，并在该模型训练结束后使用剪枝。 
    * load：加载已有模型参数，并在该模型上进行剪枝。 
* model_path：当 ``--mode`` 参数为'load'时指定从哪里读取模型参数
* pruning_nodes：参加剪枝的神经元节点下标，该下标仅考虑神经元节点

非结构化剪枝（unstructured_prune）
-----------------------------------------
介绍：
    该APP的方法来源于论文《Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework》，它提出了一种基于卷积核活动水平的结构化剪枝方法，
    称为基于脉冲通道活动（SCA）的网络剪枝框架。受突触可塑性机制的启发，我们的方法通过在训练过程中剪枝和再生卷积核来动态调整网络结构，从而增强模型对当前目标任务的适应性。
    在保持模型性能的同时，该方法优化了网络架构，最终降低了计算负载并加速了推理过程。

引用：
    Zhaofei Yu et al., Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework

使用方法：

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

参数说明：

``unstructured_prune.py`` 的大部分参数与Soul训练模型的一般流程相似，如有疑问请参考 :doc:`../params`
此处仅介绍 ``unstructured_prune.py`` 的特殊参数：

* TET：损失函数是否使用TET因子
* TET_phi：TET因子的phi值
* TET_lambda：TET因子的lambda值
* save_latest：是否保存最终剪枝结果（全模型）
* not_prune_weight：是否不针对权重进行剪枝
* not_prune_neuron：是否不针对神经元进行剪枝
* finetune：是否启用微调流程
* prune_optimizer：剪枝流程使用的优化器，可选项：adam、sgd
* prune_lr：剪枝流程使用的初始学习率
* prune_weight_decay：剪枝流程使用的权重削减程度
* penalty_lmbda：损失值使用的惩罚项lmbda值
* accumulate_step：训练优化步长
* temp_scheduler：温度调度器的初始化参数，使用格式如下：

    --temp_scheduler <init temp> <final temp>
    or --temp_scheduler <init temp> <final temp> <T0> <Tmax>
    or --temp_scheduler <init temp of weight> <init temp of neuron> <final temp of weight> <final temp of neuron> <T0> <Tmax>

* finetune_lr：微调流程使用的初始学习率
* epoch_finetune：微调流程轮次
* finetune_lr_scheduler：微调使用的调度器设置，使用格式如下：

    --scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
    or --scheduler Step [minestones]

* neuron_prune_layers：考虑剪枝的神经元节点下标
* weight_prune_layers：考虑剪枝的卷积层节点下表
* mask_init_factor：剪枝mask的初始化设置，使用格式如下：

    --mask-init-factor <weights mean> <neurons mean> <weights std> <neurons std>

* criterion：损失值计算标准，可选项为：MSE、CE
