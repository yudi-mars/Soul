# ESBench

## Overview

In ESBench, we have proposed three benchmarking tasks with research questions:

- RQ1: How do neuronal behaviors and architectures impact the deployability of EdgeSNNs?
- RQ2: How effectively do basic lightweighting strategies support the deployment of EdgeSNNs?
- RQ3: How does device heterogeneity influence the performance of deployed EdgeSNNs?

For user convenience, we arange the corresponding code for each RQ in different folders without mutual inferences:

- bench1-neuronal
- bench2-lightweight
- bench3-devices

Each folder contains addition `readme.md` file to guide the user how to run corresponding results in the paper with the code.

The `sample_generator.py` is a automatic scripts to random select samples from the corresponding datasets for inference evaluation on resource-constrained edge devices, since those devices may have no SD card to store the whole test set for complete evaluation. The generated code will be stored in `sample` folder. User can copy this folder and read the interior samples (by modifying the related path in corresponding code file) for on-device testing evaluation.

After modifying the corresponding dataset directory for each dataset and , the user can run:
```
python sample_generator.py
```
to generate the expected samples.

For further benchmarking research with different RQ, please switch the workingspace to corresponding `bench*` folder, e.g., RQ1 for:
```
cd bench1-neuronal
```

## Dataset Information

The dataset description of the selected vision datasets in this paper is provided in **Appendix B.2**.

Here, we provide the download link for all selected datasets:

- [CIFAR10/100](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf) [[Download Link](https://www.cs.toronto.edu/~kriz/cifar.html)]
- [Tiny-ImageNet](https://ieeexplore.ieee.org/abstract/document/5206848/) [[Download Link](https://www.kaggle.com/c/tiny-imagenet)]
- [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) [[Download Link](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)]
- [DVS-Gesture](https://ieeexplore.ieee.org/document/8100264) [[Download Link](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)]


As we have mentioned in the main content, ESBench use vision tasks as an example to benchmark the on-device application for SNNs, since most SNN-related research use this modality to demonstrate the efficiency of SNNs. We leave datasets with other modality as future work. 

_Note: Current ESBench is still underdevelopment and we will continually improve this repository._

## More details about energy cost statistics

As ESBench is a benchmarking study for practical deployments, we prefer reporting the real value for each metrics (accuracy, energy cost, and latency). As Jetson serial devices are all embedded with power monitoring sensors, they can report the practical energy cost for the whole on-device inference process. 

### Energy Efficiency of SNN: A Hardware View

We encourage reviewers who are less familiar with the SNN domain or energy-related evaluations to refer to this section for a more detailed explanation of the energy cost measurement methodology. This section will provide valuable context for understanding the aspects of energy consumption discussed in the main text.

Neuromorphic chips are widely recognized for their energy-efficient properties, offering a promising alternative for computation in specific domains. 
While the Von Neumann architecture remains the dominant paradigm in most modern computing devices, it faces significant challenges when applied to computation- and data-intensive neural network applications. One critical issue is the _memory wall_ problem, characterized by the inefficiency and high energy consumption associated with frequent data movement between memory and processing units. This limitation underscores the need for innovative architectures, such as neuromorphic computing, which inherently address these inefficiencies by closely integrating memory and computation.

In recent decades, researchers have drawn inspiration from biological neurons to design architectures that address the limitations of neural network applications, leading to the development of non-Von Neumann architectures and neuromorphic chips. Prominent examples include Tianjic, TrueNorth, Loihi, and Darwin3, which leverage crossbar architectures to emulate the parallel computation of the human brain and accelerate SNN calculations. These architectures organize processing units in a grid-like x-y coordinate system, where each core contains multiple \textit{neurons} functioning as Process-in-Memory (PIM) units. By closely aligning the speeds of memory storage and accumulation operations within each neuron, these designs effectively mitigate Von Neumann architectures' data movement (_memory wall problem_) issue, enabling more energy-efficient and scalable computation.

Another significant factor contributing to the energy efficiency of neuromorphic chips is the spike-sparsity property inherent to SNNs. A fundamental principle of neuromorphic design is that a neuron's axon and dendrite components within the chip are active only when spikes are transmitted; otherwise, they remain in a low-power idle state. This behavior ensures that most chip components consume minimal energy during operation. Additionally, current neuromorphic chip designers are investigating using Memory Resistors (Memristors) as a foundational circuit material. Memristors adjust their resistance based on the movement of ions or charges within the material, closely mirroring the biological characteristics of neurons, further enhancing the energy efficiency and bio-mimetic functionality of these chips.

When discussing the energy consumption of SNNs, we typically refer to both training and inference energy consumption. 
Regarding training energy consumption, the most effective approach for training SNNs currently involves executing surrogate-gradient-based BPTT on GPUs. While this method enables efficient and effective SNN training, its energy consumption scales with the number of time steps compared to training ANNs of the same scale. However, training operations are typically assumed to occur in resource-unconstrained scenarios, such as during device charging. Therefore, training energy consumption is not the primary focus of our concerns. In resource-constrained scenarios, the priority of energy consumption considerations increases, highlighting the energy-efficient advantages and research significance of SNNs.
**That's why researchers focus on evaluating and reporting the inference computational cost of SNNs in SNN-related studies.**
As previously mentioned, the energy-saving properties of SNNs can only be effectively realized on neuromorphic chips. Although directly running SNNs does not save energy for conventional devices, the computational cost (complexity) of different SNN models during inference can still be estimated. Consequently, inference energy consumption is often theoretically approximated by evaluating computational complexity.


### Theoretical Energy Cost

Due to the device heterogeneity, however, not all the devices are embedded with such sensors. Hence, when answering the RQ3 across different device platform, we choose the commonly-used theoretical energy cost counting measures to compare different on-device implementations, which _counting the number of total computing operation for the SNN model implementing inference per sample_.

For SNNs, the theoretical energy consumption of layer $l$ in SNNs can be calculated as: 
$$
    E(l) = E_{AC}\times \mathrm{SOPs}(l)
$$
where $\mathrm{SOPs}$ are the number of spike-based accumulate (AC) operations (a.k.a. synaptic operations).

For traditional ANNs, the theoretical energy consumption required by a specific layer $b$ can be estimated by:
$$
    E(b) = E_{MAC}\times \mathrm{FLOPs}(b)
$$
where $\mathrm{FLOPs}$ is the floating point operations of $b$, representing the number of multiply-and-accumulate (MAC) operations. Following other SNN-related research, 
we also assume that the MAC and AC operations are implemented on the 45nm hardware \cite{horowitz20141}, where $E_{MAC} = 4.6\mathrm{pJ}$ and $E_{AC} = 0.9\mathrm{pJ}$, where $1\mathrm{J}=10^{3}\mathrm{mJ}=10^{12}\mathrm{pJ}$.
The number of SOPs at the layer $l$ of SNN is estimated as 
$$
    \mathrm{SOPs}(l) = T \times \gamma \times \mathrm{FLOPs}(l)
$$
where $T$ is the number of times steps during simulation, and $\gamma$ is the firing rate of the input spike train of the layer $l$.

Therefore, we can approximate the total theoretical energy consumption for SNNs and ANNs layer by layer. Future advancements in neuromorphic hardware are expected to decrease energy consumption further.
