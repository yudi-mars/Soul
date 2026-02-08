Gradient Surrogate Functions
============

The neuron computation process involves threshold filtering operations. To prevent these non-differentiable operations from affecting model training, we use different gradient surrogate functions for different neurons (in accordance with the original papers).
If you want to experiment with gradient surrogate functions, you can modify the ``surrogate`` attribute in the corresponding ``config/neuron/*.yaml`` configuration files to achieve this.

atan
----------------
介绍：
    Arctangent gradient surrogate function
引用：
    Yanqi Chen et al., "State transition of dendritic spines improves learning of sparse spiking neural networks", ICML'2022.

erf
----------------
介绍：
    Gaussian error function (erf) surrogate spike function
引用：
    Esser S. K. et al., "Backpropagation for energy-efficient neuromorphic computing", NeurIPS'2015.

sigmoid
----------------
介绍：
    Sigmoid gradient surrogate function
引用：
    Woźniak S. et al., "Deep learning incorporating biologically inspired neural dynamics and in-memory computing", Nat. Mach. Intell.'2020.

rectangle
----------------
介绍：
    Rectangular surrogate spike function
引用：
    Yujie Wu et al., "Direct Training for Spiking Neural Networks: Faster, Larger, Better", AAAI'2019.

fastsigmoid
----------------
介绍：
    Fast sigmoid gradient surrogate function
引用：
    Woźniak S. et al., "Deep learning incorporating biologically inspired neural dynamics and in-memory computing", Nat. Mach. Intell.'2020.

qpseudospike
----------------
介绍：
    q-th order pseudo-spike (q-PseudoSpike) surrogate spike function
引用：
    Herranz-Celotti, L. et al., "Stabilizing Spiking Neuron Training", NeurIPS'2015.

softsign
----------------
介绍：
    Softsign surrogate spike function
引用：
    Zenke F. et al., "Superspike: Supervised learning in multilayer spiking neural networks", Neural computation'2018.

quandratic
----------------
介绍：
    Piecewise quadratic surrogate spike function
引用：
    Bellec, G. et al., "Long short-term memory and learning-to-learn in networks of spiking neurons", NeurIPS'2018.

exp
----------------
介绍：
    Piecewise exponential gradient surrogate function
引用：
    Shrestha, S. B. et al., "Slayer: Spike layer error reassignment in time", NeurIPS'2018.

superspike
----------------
介绍：
    Superspike gradient surrogate function
引用：
    Zenke F. et al., "Superspike: Supervised learning in multilayer spiking neural networks", Neural computation'2018.

ternary
----------------
介绍：
    Specific gradient surrogate function for Temporal Leaky Integrate-and-Fire (TLIF)
引用：
    Yufei Guo et al., "Ternary spike: Learning ternary spikes for spiking neural networks", AAAI'2024.

quant
----------------
介绍：
    Specific gradient surrogate function for Temporal Nonlinear Leaky Integrate-and-Fire (INTLIF)
引用：
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025.

quant4
----------------
介绍：
    Specific surrogate function for Temporal Nonlinear Leaky Integrate-and-Fire (INTLIF)
引用：
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025.