Basic Concepts
========
Author of this tutorial:  `Tecl Cajol1e <https://github.com/cajol1e>`_

To better understand the usage of the Soul toolkit, we will first introduce some basic concepts about Spiking Neural Networks (SNNs).
Soul is an open-source toolkit based on Python and PyTorch for building Spiking Neural Network (SNN) applications.Users of the 
SpikingJelly framework should first be familiar with PyTorch. If users are not well-versed in PyTorch,we recommend learning the 
PyTorch Basic Tutorials <https://pytorch.org/tutorials/>_ first.

Spiking Neural Network
-------------------------------------------
Spiking Neural Network (SNN) is the third generation of artificial neural networks, with the core of highly simulating the working 
mechanism of neurons in the biological brain.Its neurons integrate input signals to accumulate membrane potential, and when the 
potential reaches a threshold, discrete spike signals are emitted.Signals are transmitted through synapses, and synaptic strength 
can be adjusted through plasticity to achieve learning.It has the core advantages of low energy consumption and accurate processing 
of temporal information, with a much higher bionic degree than traditional neural networks. Currently, it has broad application 
prospects in fields such as brain-computer interfaces,low-power intelligent hardware, and autonomous driving.


Neuron
-------------------------------------------
The neuron of a Spiking Neural Network is the core computing unit that simulates the structure and function of biological neurons, 
and is the foundation of SNN information processing.It receives spike inputs from other neurons and integrates membrane potential;
when the accumulated membrane potential reaches a preset threshold, it emits discrete spike signals,and the potential resets rapidly 
after emission. Meanwhile, it can connect with other neurons through synapses, and the connection strength can be dynamically 
adjusted according to spike emission characteristics,thereby realizing learning functions.For introductions and configuration 
information about neurons, please refer to :doc:`../neuron`

Gradient Surrogate Function
-------------------------------------------
Gradient surrogate in Spiking Neural Network training is a core method to solve the problem of gradient vanishing caused by the 
discreteness of neuron spike emission, which makes direct backpropagation impossible.It replaces the discrete spike emission function 
of neurons with a continuous and differentiable approximation function (such as Sigmoid, Softplus functions),thereby calculating 
approximate gradients, enabling optimization algorithms of traditional deep learning to be applied to SNN training, balancing training 
feasibility and bionic characteristics.For introductions to gradient surrogate functions, please refer to :doc:`../gradient`

Encoding Method
-------------------------------------------
The encoding method of Spiking Neural Networks, at its core, converts continuous input signals into discrete neuronal spike sequences, 
which is a crucial preprocessing step for SNNs to process information. Common types mainly include the following:
* Rate Coding: The most commonly used method, which characterizes input intensity by the spike firing frequency of neurons within a 
fixed time window. The stronger the input, the higher the spike frequency. It is simple to implement but ignores temporal details.
* Temporal Coding: Transmits information through the precise timing of spike firing. For example, the stronger the input, the earlier 
the spike is fired. It can accurately capture the temporal characteristics of signals and has a higher bionic degree.
* Phase Coding: Encodes information through the phase difference of spikes relative to a reference signal, suitable for processing periodic input signals.
* Spike Count Coding: Directly maps the total number of spikes within a fixed time window to the magnitude of the input signal. It has intuitive logic and is easy to interface with traditional models.
For introductions to encoding methods, please refer to :doc:`../encoding`

Dataset
-------------------------------------------
Datasets for SNNs are built around spike sequences and temporal events, divided into two categories: native spike datasets and converted spike datasets, adapted to their event-driven and temporal processing characteristics:
* Native Spike Datasets: Directly collected from bionic sensors or biological nervous systems, naturally compatible with SNNs.
* Converted Spike Datasets: Converted from classic static datasets of ANNs, transforming continuous values into spike sequences through methods such as rate coding and temporal coding.
For introductions and configuration information about datasets, please refer to :doc:`../datasets/index`