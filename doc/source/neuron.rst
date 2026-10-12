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
-----------------
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
    只支持多步模式，输入的时间步数须等于 time_step
引用：
    Xingting Yao et al., "GLIF: A Uniﬁed Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", NeurIPS'2022. https://github.com/CAS-CLab/Gated-LIF
配置：

* surrogate：反向传播的替代函数
* init_tau：膜时间常数的初始值
* init_v_threshold：神经元发放阈值的初始值
* init_conduct：电导的初始值
* init_linear_decay：线性泄漏的初始值，缺省为 init_v_threshold / (2 * time_step)
* init_v_subreset：减性重置量的初始值，缺省为 init_v_threshold

INTLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
    发放量化后的整数值脉冲，输出 spike / norm。默认以单步模式运行：整段输入作为一个时间步积分，即各时间步独立量化；
    在配置中设置 step_mode: m 则沿时间维逐步积分
引用：
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025. https://github.com/BICLab/Spike2Former
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* v_reset：膜电位的初值（重置总是减去 v_threshold 的软重置）
* detach_reset：是否将重置操作从计算图中分离
* tau：膜电位衰减的时间常数
* norm：输出脉冲的归一化系数

ParallelSpikingNode
-------------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
    无状态，一次处理整段序列，输入的时间步数须等于 time_step
引用：
    Wei Fang et al., "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies", NeurIPS'2023. https://github.com/fangwei123456/Parallel-Spiking-Neuron
配置：

* surrogate：反向传播的替代函数

TLIFNode
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
    发放 {-1, 0, 1} 三值脉冲，输出 spike / 8；膜电位不泄漏，其余行为（包括默认的单步模式）同 INTLIFNode
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
    阈值可学习，无状态，每次调用都从零状态处理整段输入序列
引用：
    Siqi Wang et al., "LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout", NeurIPS'2022. https://github.com/sq117/LTMD
配置：

* kappa：LTM-D模型的kappa值

STBIF
----------------
介绍：
    脉冲神经网络（SNN）的泄漏积分 - 发放（LIF）变体神经元实现
    按 q_threshold 归一化后的电荷达到 1 时发放并减 1，反向传播时由替代函数提供梯度
引用：
    Kang You et al., "VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network", CVPR'2025. https://github.com/Intelligent-Computing-Research-Group/ViStream
配置：

* surrogate：反向传播的替代函数
* q_threshold：量化阈值
* level：量化等级
* detach_reset：是否将重置操作从计算图中分离（可选，缺省为 True）

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
* alpha：每次发放后动态阈值的放大系数

IzhikevichNode
----------------
介绍：
    脉冲神经网络（SNN）的 Izhikevich 神经元实现，带有适应电流
引用：
    E. M. Izhikevich, "Simple model of spiking neurons," in IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003.
配置：

* tau：膜时间常数
* v_c：二次项中的临界电压
* a0：二次项系数
* v_threshold：神经元的发放阈值
* v_reset：膜电位的重置值
* v_rest：静息膜电位
* w_rest：适应电流的初值
* tau_w：适应电流的时间常数
* a：适应电流的亚阈值耦合强度
* b：每次发放后适应电流的增量
* surrogate：反向传播的替代函数
* detach_reset：是否将重置操作从计算图中分离
* step_mode：推理模式，m 为多步，s 为单步
* store_v_seq：是否存储膜电位序列

DGNNode
--------------
介绍：
    动态门控神经元（Dynamic Gated Neuron）：膜电位的衰减系数由随输入变化的动态电导门控
    在通用模型中按 s-DGN 运行（所有突触共享可学习的平衡电位 E）；lenet_dgn 为每个突触层增加电导支路，即完整的 DGN
引用：
    Qianyi Bai et al., "A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks", ICLR'2026.
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* g_l：漏电导，膜电位的衰减系数为 sigmoid(1 - g_l - 电导输入)
* tau_s：突触电流迹的时间常数，~ 表示不做突触滤波
* init_equilibrium：共享平衡电位 E 的初值，只在没有电导输入时使用
* v_reset：膜电位的重置值，~ 表示软重置
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列

DLIFNode
--------------
介绍：
    树突 LIF 神经元：胞体为 LIF 动力学，树突对输入通道做两两相乘的双线性整合（不含自身平方项）
    树突整合由 lenet_dlif 中的 DendriticBilinear 层完成，在其他模型中等同于下述参数的 LIF
配置：

* v_threshold：神经元的发放阈值
* surrogate：反向传播的替代函数
* tau：膜时间常数
* decay_input：是否采用衰减输入
* v_reset：膜电位的重置值，~ 表示软重置
* detach_reset：是否将重置操作从计算图中分离
* store_v_seq：是否存储膜电位序列

POLARANode
--------------
介绍：
    极化感知神经元：用时间核函数描述去极化、复极化与超极化，没有发放后的重置；阈下膜电位被随机放大，阈值逐步自适应
    发放的脉冲取 0 或 [0.5, 1) 内的连续值
引用：
    Matthew Lai and Longbing Cao, "Stabilizing Spiking Neurons Through Biologically Inspired Polarization", AAAI'2026.
配置：

* v_threshold：发放阈值的初值，之后逐步自适应
* surrogate：反向传播的替代函数
* A：膜电位衰减核的幅度
* alpha：膜电位衰减核的衰减速率
* window：时间核的窗口长度
* noise：阈下随机放大的幅度，0 表示不加噪声
* store_v_seq：是否存储膜电位序列
