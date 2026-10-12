Neurons
=======

Neurons in Spiking Neural Networks (SNNs) are the core units for information processing and transmission in the network. Essentially, 
they simulate the spatiotemporal dynamic characteristics of biological neurons—by receiving and integrating input signals,
they emit spikes when specific conditions are met, and transmit spike signals to downstream neurons, enabling information representation 
and computation based on temporal coding. This is fundamentally different from continuous-value neurons in traditional Artificial Neural 
Networks (ANNs).
Next, we will separately explore different neurons in Soul, as well as how to use and adjust them.

LIFNode
----------------
Introduction:
    Implementation of the leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023. https://github.com/fangwei123456/spikingjelly
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* tau：Membrane time constant
* decay_input：Whether to adopt decaying input
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence

ParametricLIFNode
-----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Wei Fang et al., "Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks", ICCV'2021. https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* init_tau： Initial value of the membrane time constant
* decay_input：Whether to adopt decaying input
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence

CLIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Yulong Huang et al., "CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", ICML'2024. https://github.com/HuuYuLong/Complementary-LIF
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* tau：Membrane time constant
* decay_input：Whether to adopt decaying input
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence

GatedLIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
    Multi-step mode only; the number of input time steps must equal time_step
Citation:
    Xingting Yao et al., "GLIF: A Uniﬁed Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", NeurIPS'2022. https://github.com/CAS-CLab/Gated-LIF
Configurations：

* surrogate：Surrogate function for backpropagation
* init_tau：Initial value of the membrane time constant
* init_v_threshold： Initial value of the neuron's firing threshold
* init_conduct： Initial value of conductance
* init_linear_decay：Initial value of linear decay, init_v_threshold / (2 * time_step) by default
* init_v_subreset： Initial value of sub-reset membrane potential, init_v_threshold by default

INTLIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
    Fires quantized integer spikes and outputs spike / norm. It runs in single-step mode by default: the whole input is integrated
    as one time step, i.e. every time step is quantized independently; set step_mode: m to integrate along the time dimension
Citation:
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025. https://github.com/BICLab/Spike2Former
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* v_reset：Initial membrane potential (the reset always subtracts v_threshold)
* detach_reset：Whether to detach the reset operation from the computation graph
* tau：Decay time constant of the membrane potential
* norm：Normalization factor of the output spikes

ParallelSpikingNode
-------------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
    Stateless, processes the whole sequence at once; the number of input time steps must equal time_step
Citation:
    Wei Fang et al., "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies", NeurIPS'2023. https://github.com/fangwei123456/Parallel-Spiking-Neuron
Configurations：

* surrogate：Surrogate function for backpropagation

TLIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
    Fires ternary spikes {-1, 0, 1} and outputs spike / 8; the membrane potential does not leak, otherwise (including the default
    single-step mode) it behaves like INTLIFNode
Citation:
    Yufei Guo et al., "Ternary spike: Learning ternary spikes for spiking neural networks", AAAI'2024. https://github.com/yfguo91/Ternary-Spike
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph

IELIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Xuerui Qiu et al., "Quantized Spike-Driven Transformer", ICLR'2025. https://github.com/bollossom/QSD-Transformer
Configurations：

* lens：Bit-width parameter
* surrogate：Surrogate function for backpropagation

LTMD
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
    Learnable threshold, stateless: every call processes the whole input sequence from a zero state
Citation:
    Siqi Wang et al., "LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout", NeurIPS'2022. https://github.com/sq117/LTMD
Configurations：

* kappa：Kappa value of the LTM-D model

STBIF
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
    Fires when the charge normalized by q_threshold reaches 1 and then subtracts 1; gradients come from the surrogate function
Citation:
    Kang You et al., "VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network", CVPR'2025. https://github.com/Intelligent-Computing-Research-Group/ViStream
Configurations：

* surrogate：Surrogate function for backpropagation
* q_threshold： Quantization threshold
* level：Quantization level
* detach_reset：Whether to detach the reset operation from the computation graph (optional, True by default)

ILIFNeuron
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Zhenxin Lei et al., "ILIF: Temporal Inhibitory Leaky Integrate-and-Fire Neuron for Overactivation in Spiking Neural Networks", IJCAI'2025. https://github.com/kaisun1/ILIF
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* tau：Membrane time constant
* decay_input：Whether to adopt decaying input
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence

RPLIFNode
----------------
Introduction:
    Implementation of the Refractory Period Leaky Integrate-and-Fire (RPLIF) neuron
Citation:
    Li, Yang, et al., "Incorporating the Refractory Period into Spiking Neural Networks through Spike-Triggered Threshold Dynamics", MM'2025. https://arxiv.org/pdf/2509.17769
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* tau：Membrane time constant
* decay_input：Whether to adopt decaying input
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence
* alpha：Scaling factor of the dynamic threshold after each spike

IzhikevichNode
----------------
Introduction:
    Implementation of the Izhikevich neuron with an adaptation current for Spiking Neural Networks (SNNs)
Citation:
    E. M. Izhikevich, "Simple model of spiking neurons," in IEEE Transactions on Neural Networks, vol. 14, no. 6, pp. 1569-1572, Nov. 2003.
Configurations：

* tau：Membrane time constant
* v_c：Critical voltage of the quadratic term
* a0：Coefficient of the quadratic term
* v_threshold：Firing threshold of the neuron
* v_reset：Reset value of the membrane potential
* v_rest：Resting membrane potential
* w_rest：Initial value of the adaptation current
* tau_w：Time constant of the adaptation current
* a：Subthreshold coupling of the adaptation current
* b：Increment of the adaptation current after each spike
* surrogate：Surrogate function for backpropagation
* detach_reset：Whether to detach the reset operation from the computation graph
* step_mode：Step mode, 'm' for multi-step and 's' for single-step
* store_v_seq：Whether to store the membrane potential sequence

DGNNode
--------------
Introduction:
    Dynamic Gated Neuron: the decay of the membrane potential is gated by an activity-dependent dynamic conductance
    It runs as s-DGN (all synapses share a learnable equilibrium potential E) in the general models; lenet_dgn adds a conductance pathway to every synaptic layer, i.e. the full DGN
Citation:
    Qianyi Bai et al., "A Brain-Inspired Gating Mechanism Unlocks Robust Computation in Spiking Neural Networks", ICLR'2026.
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* g_l：Leak conductance; the decay factor of the membrane potential is sigmoid(1 - g_l - conductance input)
* tau_s：Time constant of the synaptic current traces, ~ to disable the traces
* init_equilibrium：Initial value of the shared equilibrium potential E, only used without a conductance input
* v_reset：Reset value of the membrane potential, ~ for soft reset
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence

DLIFNode
--------------
Introduction:
    Dendritic LIF neuron: LIF dynamics at the soma, while the dendrites integrate pairwise products of the input channels (without self terms)
    The dendritic integration is done by the DendriticBilinear layers of lenet_dlif; in other models it is a LIF neuron with the settings below
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* tau：Membrane time constant
* decay_input：Whether to adopt decaying input
* v_reset：Reset value of the membrane potential, ~ for soft reset
* detach_reset：Whether to detach the reset operation from the computation graph
* store_v_seq：Whether to store the membrane potential sequence

POLARANode
--------------
Introduction:
    Polarization-aware neuron: depolarization, repolarization and hyperpolarization are described by temporal kernels and there is no reset after firing; sub-threshold potentials are stochastically amplified and the threshold adapts at every step
    The spikes are 0 or continuous values in [0.5, 1)
Citation:
    Matthew Lai and Longbing Cao, "Stabilizing Spiking Neurons Through Biologically Inspired Polarization", AAAI'2026.
Configurations：

* v_threshold：Initial firing threshold, adapted at every time step
* surrogate：Surrogate function for backpropagation
* A：Magnitude of the membrane decay kernel
* alpha：Rate of the membrane decay kernel
* window：Length of the temporal kernel window
* noise：Amplitude of the stochastic sub-threshold amplification, 0 to disable
* store_v_seq：Whether to store the membrane potential sequence
