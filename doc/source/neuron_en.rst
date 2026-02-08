Neurons
======

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
----------------
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
Citation:
    Xingting Yao et al., "GLIF: A Uniﬁed Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks", NeurIPS'2022. https://github.com/CAS-CLab/Gated-LIF
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* init_tau：Initial value of the membrane time constant
* init_v_threshold： Initial value of the neuron's firing threshold
* init_conduct： Initial value of conductance
* init_linear_decay：Initial value of linear decay
* init_v_subreset： Initial value of sub-reset membrane potential

INTLIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Zhenxin Lei et al., "Spike2former: Efficient spiking transformer for high-performance image segmentation", AAAI'2025. https://github.com/BICLab/Spike2Former
Configurations：

* v_threshold：Firing threshold of the neuron
* surrogate：Surrogate function for backpropagation
* v_reset：Reset value of the membrane potential
* detach_reset：Whether to detach the reset operation from the computation graph

ParallelSpikingNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Wei Fang et al., "Parallel Spiking Neurons with High Efficiency and Ability to Learn Long-term Dependencies", NeurIPS'2023. https://github.com/fangwei123456/Parallel-Spiking-Neuron
Configurations：

* surrogate：Surrogate function for backpropagation

TLIFNode
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
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
Citation:
    Siqi Wang et al., "LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout", NeurIPS'2022. https://github.com/sq117/LTMD
Configurations：

* steps：Time step for running the LTM-D model
* kappa：Kappa value of the LTM-D model

STBIF
----------------
Introduction:
    Implementation of a variant leaky integrate-and-fire (LIF) neuron for Spiking Neural Networks (SNNs)
Citation:
    Kang You et al., "VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network", CVPR'2025. https://github.com/Intelligent-Computing-Research-Group/ViStream
Configurations：

* surrogate：Surrogate function for backpropagation
* q_threshold： Quantization threshold
* level：Quantization level

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
* alpha：Scaling factor for gradient surrogate