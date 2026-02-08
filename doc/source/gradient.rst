梯度代理函数
============

神经元的计算过程中存在阈值过滤的操作，为了避免这种不可导的操作影响模型训练，我们对不同的神经元采用不同的梯度代理函数（按照原论文）。
若您想就梯度代理函数做一些尝试，可以修改对应的 ``config/neuron/*.yaml`` 配置中的 ``surrogate`` 属性来实现。

atan
----------------
介绍：
    反正切梯度代理函数
引用：
    Yanqi Chen et al., "State transition of dendritic spines improves learning of sparse spiking neural networks", ICML'2022.

erf
----------------
介绍：
    高斯误差（erf）替代脉冲函数
引用：
    Esser S. K. et al., "Backpropagation for energy-efficient neuromorphic computing", NeurIPS'2015.

sigmoid
----------------
介绍：
    Sigmoid梯度代理函数
引用：
    Woźniak S. et al., "Deep learning incorporating biologically inspired neural dynamics and in-memory computing", Nat. Mach. Intell.'2020.

rectangle
----------------
介绍：
    矩形替代脉冲函数
引用：
    Yujie Wu et al., "Direct Training for Spiking Neural Networks: Faster, Larger, Better", AAAI'2019.

fastsigmoid
----------------
介绍：
    Fast sigmoid梯度代理函数
引用：
    Woźniak S. et al., "Deep learning incorporating biologically inspired neural dynamics and in-memory computing", Nat. Mach. Intell.'2020.

qpseudospike
----------------
介绍：
    q 阶伪脉冲（q-PseudoSpike）替代脉冲函数
引用：
    Herranz-Celotti, L. et al., "Stabilizing Spiking Neuron Training", NeurIPS'2015.

softsign
----------------
介绍：
    软符号替代脉冲函数
引用：
    Zenke F. et al., "Superspike: Supervised learning in multilayer spiking neural networks", Neural computation'2018.

quandratic
----------------
介绍：
    分段二次替代脉冲函数
引用：
    Bellec, G. et al., "Long short-term memory and learning-to-learn in networks of spiking neurons", NeurIPS'2018.

exp
----------------
介绍：
    分段指数梯度代理函数
引用：
    Shrestha, S. B. et al., "Slayer: Spike layer error reassignment in time", NeurIPS'2018.

superspike
----------------
介绍：
    超脉冲梯度代理函数
引用：
    Zenke F. et al., "Superspike: Supervised learning in multilayer spiking neural networks", Neural computation'2018.

ternary
----------------
介绍：
    时序泄漏积分 - 发放（TLIF）的特定梯度代理函数
引用：
    Yufei Guo et al., "Ternary spike: Learning ternary spikes for spiking neural networks", AAAI'2024.

quant
----------------
介绍：
    时序非线性泄漏积分 - 发放（INTLIF）的特定梯度代理函数
引用：
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025.

quant4
----------------
介绍：
    时序非线性泄漏积分 - 发放（INTLIF）的特定替代函数
引用：
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025.