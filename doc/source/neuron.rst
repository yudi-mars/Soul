神经元
======

脉冲神经网络（SNN）的神经元是网络信息处理与传递的核心单元，其本质是模拟生物神经元的时空动力学特性——通过接收、整合输入信号，
在满足特定条件时发放脉冲（Spike），并将脉冲信号传递至下游神经元，实现基于时间编码的信息表征与计算，这与传统人工神经网络（ANN）中连续值神经元有本质区别。
接下来，我们将分别了解Soul中不同的神经元，以及如何使用、调整它们。

LIFNode
----------------
介绍：
    脉冲神经网络（SNN）的 leaky integrate-and-fire（LIF，泄漏积分 - 发放）神经元实现
引用：
    Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023. https://github.com/fangwei123456/spikingjelly
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* tau：膜时间常数
* decay_input：是否采用衰减输入
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列

ParametricLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Wei Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks", ICCV'2021. https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* init_tau：膜时间常数的初始值
* decay_input：是否采用衰减输入
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列

CLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Yulong Huang et al., "CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", ICML'2024. https://github.com/HuuYuLong/Complementary-LIF
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* tau：膜时间常数
* decay_input：是否采用衰减输入
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列

GatedLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Xingting Yao et al., "GLIF: A Uniﬁed Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", NeurIPS'2022. https://github.com/CAS-CLab/Gated-LIF
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* init_tau：膜时间常数的初始值
* init_v_threshold：神经元发放阈值的初始值
* init_conduct：电导的初始值
* init_linear_decay：
* init_v_subreset：

INTLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025. https://github.com/BICLab/Spike2Former
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离

ParallelSpikingNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Wei Fang et al., "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies", NeurIPS'2023. https://github.com/fangwei123456/Parallel-Spiking-Neuron
配置：

* surrogate：反向传播的替代函数

TLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Yufei Guo et al., "Ternary spike: Learning ternary spikes for spiking neural networks", AAAI'2024. https://github.com/yfguo91/Ternary-Spike
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离

IELIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Xuerui Qiu et al., "Quantized Spike-Driven Transformer", ICLR'2025. https://github.com/bollossom/QSD-Transformer
配置：

* lens：位宽参数
* surrogate：反向传播的替代函数

LTMD
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Siqi Wang et al., "LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout", NeurIPS'2022. https://github.com/sq117/LTMD
配置：

* steps：运行LTM-D模型的时间步长
* kappa：LTM-D模型的kappa值

STBIF
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Kang You et al., "VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network", CVPR'2025. https://github.com/Intelligent-Computing-Research-Group/ViStream
配置：

* surrogate：反向传播的替代函数
* q_threshold：量化阈值
* level：量化等级

ILIFNeuron
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
引用：
    Zhenxin Lei et al., "ILIF: Temporal Inhibitory Leaky Integrate-and-Fire Neuron for Overactivation in Spiking Neural Networks", IJCAI'2025. https://github.com/kaisun1/ILIF
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* tau：膜时间常数
* decay_input：是否采用衰减输入
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列

RPLIFNode
----------------
介绍：
    不应期泄漏积分 - 发放（Refractory Period LIF）神经元的实现
引用：
    Li, Yang, et al., "Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics", MM'2025. https://arxiv.org/pdf/2509.17769
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* tau：膜时间常数
* decay_input：是否采用衰减输入
* v_reset：膜电位的重置值
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列
* alpha：梯度代理的缩放因子