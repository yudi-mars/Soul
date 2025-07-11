# Soul: A Toolbox for Developing Edge Intelligence Applications with Spiking Neural Networks

<p align="center">
    <img src="./assets/code-logo.jpg" width="50%">
</p>

--------------------------------------------------------------------------------

*“I have always been convinced that the only way to get artificial intelligence to work is to do the computation in a way similar to the human brain.”——Geoffrey Hinton*

<p align="center">
  <a href="#Overview">Overview</a> •
  <!-- <a href="#Feature">Feature</a> • -->
  <a href="#how-to-run">How to Run</a> •
  <!-- <a href="#dataset-support">Dataset Support</a> • -->
  <a href="#Results-Visualization">Results Visualization</a> •
  <a href="#documentation">Documentation</a> •
  <!-- <a href="#license">License</a> • -->
  <!-- <a href="#Acknowledgments">Acknowledgments</a> •
  <a href="#Contact">Contact</a> -->
</p>

SOUL (**S**NN-based **O**pen so**U**rce too**L**kit) is developed based on Python and PyTorch for reproducing and developing SNN-based brain-inspired computing algorithms in a unified, comprehensive, and efficient framework for research purposes and practical deployment at the edge.

## Overview

TBD 一张图 @yudi

### Feature

TBD 

- feature 1
- feature 2

### Dataset Support

1. Vision Sensing
    - [CIFAR10/100](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf) [[Download Link](https://www.cs.toronto.edu/~kriz/cifar.html)]
    - [Tiny-ImageNet](https://ieeexplore.ieee.org/abstract/document/5206848/) [[Download Link](https://www.kaggle.com/c/tiny-imagenet)]
    - [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) [[Download Link](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)]
    - [DVS-Gesture](https://ieeexplore.ieee.org/document/8100264) [[Download Link](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)]

2. Motion Sensing
    - [UCI](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930) [[Download Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)]
    - [HHAR](https://dl.acm.org/doi/10.1145/2809695.2809718) [[Download Link](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)] *Note: Only Phone-related data is used in this repository*
    - [MotionSense](https://dl.acm.org/doi/10.1145/3302505.3310068) [[Download Link](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset)]
    - [Shoaib](https://www.mdpi.com/1424-8220/14/6/10146) [[Download Link](https://www.utwente.nl/en/eemcs/ps/dataset-folder/ut-data-complex.rar)]

3. Acoustic Sensing
    - [Google Speech Commands](https://arxiv.org/abs/1804.03209) [[Download Link](https://www.researchgate.net/publication/266384007_Sensors_Activity_Recognition_DataSet)]
    - [UrbanSound](https://dl.acm.org/doi/10.1145/2647868.2655045) [[Download Link](https://urbansounddataset.weebly.com/)]
    - TODO @changze
    - TODO @changze

模型的文献参考

1. motion sensing (设备内的陀螺仪、磁针等采集的): UCI HAR, Opportunity（待定）/MotionSense, Shoaib, HHAR | model: SenseHAR, DCNN, spikeBIOT/HART


    
- Deep convolutional neural networks on multi-channel time series for human activity recognition (IJCAI'15)
- SenseHAR: a robust virtual activity sensor for smartphones and wearables (Sensys'19)
- BIOT: Biosignal Transformer for Cross-data Learning in the Wild (NeurIPs'23)
- Lightweight transformers for human activity recognition on mobile devices （arxiv备选）
   
2. vision sensing (一般指的是各种照相机：rgb camera，~~depth camera~~， dvs camera应该也算这里)

**新、sota且典型模型都已经在soul里实现了，我们数据集有cifar10, cifar100, imagenet, dvs-gesture, cifar10-dvs,模型有msresnet,sewresnet,vgg系列,qkformer,spikformer,metaspikeformer,spikingresformer**

3. acoustic sensing(麦克风、扬声器等收集):(数据集以keyword/event detection、emotion recognition为例?) GSC(google speech recognition), UrbanSound8k这两个肯定可以算 ｜模型有哪些？

_TBD @Changze_


## How to Run
### Command in Console 

- Running `Soul` with a single GPU in default settings
    ```shell
    CUDA_VISIBLE_DEVICES=[GPU ID] python run_soul.py
    ```

- Running `Soul` with multiple GPUs in default settings
    ```shell
    CUDA_VISIBLE_DEVICES=[GPU ID1],[GPU ID2],... torchrun --nproc_per_node=[Number of used GPU] run_soul.py
    ```

### Shell scripts

- Running `Soul` with a single GPU in default settings
    ```shell
    ./scripts/train_single.sh
    ```
    
- Running `Soul` with multiple GPUs in default settings
    ```shell
    ./scripts/train_ddp.sh
    ```

## Results Visualization

TBD 终版前给一个结果展示，比如ESBench的一些结果列上去

## Documentation

TBD 后面学生来做这个部分的事情 (Sphinx Documentation)

## TODO List

_Note: 应用以主流edge applicaiton为主(老三样), 但是举例一些特殊的应用，如果有开源数据的话 就跑一跑结果调通验证未来可行性_

### 工程任务

- [x] 2025.07.07-2025.07.14 @yudi data.py的功能模块需要拆分，分数据集进行处理会不会更好?
- [x] TBD ~~neuron对于不同场景会不会有特殊化的处理？@changze 这个需要商讨下~~ 以TS-LIFw为例，用case的形式展示如何利用Soul构建特殊lif进行特殊数据集上的应用即可
- [x] 2025.07.03-2025.07.05 @yudi run_soul独立训练、推理、算各类指标的接口,提供一些单一功能的执行文件
- [x] ~~2025.07.05-TBD @wentao 目前count_sops还无法对transformer相关模型进行统计(或者统计很慢), 需要优化调整~~
- [x] 2025.07.11-2025.07.15 @yudi 为DVS数据加载添加augmentation功能
- [ ] 2025.07.02-2025.07.15 Speech Recognition相关文献调研、模型以及数据加载复现以及融入Soul的难点评估 @changze
- [ ] 2025.07.10-2025.07.31 @changze @yudi 我们需要ANN2SNN（为vgg，resnet，spikformer至少挑选一个对应可行的资源消耗没那么严重的ANN2SNN训练方法）**从cloud service的视角展示可行性，最后cloud端给边缘端发送模型** / offload with incremental learning看看有没有可能
- [ ] 2025.07.20-2025.07.31 @changze @yudi lightweight的方法: ANN->SNN的蒸馏方法~2023cvpr 为vgg，resnet，spikformer提供一个可行的蒸馏方案 **与ANN2SNN同理**
- [ ] 2025.07.20-2025.07.31 @changze lightweight的方法: **NAS**: 如果目前只支持vgg，那就先只支持直通网络,~~但是要确认一些别的方法（比如乐透奖Panda 2022那篇ECCV）看看哪种方法开销更小~~**主要展示的是当前这个NAS方向是如何和我们的Soul融合适配的！**  example(case)导向
- [ ] 2025.07.25-2025.07.31 @changze @yudi quantization接口  **主要展示的是当前这个prune方向是如何和我们的Soul融合适配的！**
- [ ] 2025.07.11-2025.07.31 @yudi @helin prune：以example的形式展示，不是融进Soul里，针对GPU系列的设备提供一个可行方案，如structure pruning，以VGG直通网络为例，支持GPU上优化~~neuromorphic~~， ~~unstructure pruning支持neuromorphic （目前提供GPU模拟即可）~~ **主要展示的是当前这个prune方向是如何和我们的Soul融合适配的！**
- [ ] 2025.07.02-2025.07.15 Human Activity Recognition相关模型，数据加载模块融入Soul @yudi
- [x] 2025.06.23-2025.07.02 @changze Potential Modulation类目下的神经元建模，能融合的都要融合进来
    - ST-BIF [67] 找作者要到了代码，已完成
    - PLIF[74] 已经完成
    - ~~SRIF[76] guoyufei的，发现无法“即插即用”，需要指定input dim和output dim~~
    - LTMD [76] 已实现
    - ~~BDETT [77] 未开源~~
    - ~~DTA-TTFS [78] 虽然找作者要到了代码，但这个神经元的传递强制需要前一层权重的weight sum以及activation，不能在我们的文件中直接forward。淘汰~~
    - CLIF[79] 已经完成
    - ~~ABN [80] 未开源~~
    - ~~DA-LIF [81] 未开源~~
    - ~~Smooth LIF [82] 未开源~~
- [ ] 2025.12.01-TBD **终版要为每个文件添加一定的注释信息，包含输入输出代码说明，代码引用，以及必要的参考文献** 代码内参考文献可以用spikingvgg.py开头的示例为参考 !!!!!!
- [ ] TBD 展示一些spiking的LLM在一些通用设备上运行的结果(spikeGPT为例)，展示EdgeLLM future direction.(作为一个application) toy-example

### 科研任务
- [ ] 2025.07.31-TBD @Lvchangze @Yudi **设计一个针对enrich data的rehearsal-based的增量学习算法（simple but effective）**
- [ ] 2025.07.10-2025.08.31 ~~STDP训练方法调研 提供一个好用的方案，目前可以给个GPU方案，但需要给出neuromorphic chip的部署基本逻辑(STDP的特色，至少是可以避免一定的人工打标签的模型无监督更新，存在一定的特色)~~经讨论确认采用snn-ncg项目对于STDP的应用范式，需要@changze全力推进代码方针工作以及相应的难点延伸, @yudi 负责辅助、代码复审
- [x] ~~2025.07.01-2025.08.31 @yudi  temporal learning 提供一个好用的方案，可以GPU方针，但需要给出neuromorphic chip的部署基本逻辑~~tempporal learning这个方向本身只是训练方法，内在要求其实和ANN差别不大，无法完全彰显SNN的特色，哪怕是有能在neuromorphic上高效运行的潜能，也并不代表其可完全替代或补全ANN的对应功能

## Cite

```
TBD
```

## License

Please check the [Apache License](./LICENSE) that is listed in this repository.

## Acknowledgments

We want to express our sincere gratitude to the developers and contributors of the following open-source projects, which have greatly inspired and supported this work:

- [snntorch](https://github.com/jeshraghian/snntorch): Provided excellent modular design patterns for surrogate functions and input encoding schema that influenced our architecture.
- [SpikingJelly](https://github.com/fangwei123456/spikingjelly): Served as a reference for implementing spiking neuron and loading DVS dataset.
- [PyCIL](https://github.com/LAMDA-CL/PyCIL): Provided a clear and extensible data loading module for conventional dataset, which served as a valuable reference for our dataset handling.

Their valuable work laid the foundation for some key components of our project.
We also appreciate the broader open-source community for fostering a collaborative and innovative ecosystem for EdgeSNN.

## Contact

If there are any questions, please feel free to propose new features by opening an issue or contact with the author: **Di Yu**([yudi2023@zju.edu.cn](mailto:yudi2023@zju.edu.cn)), **Changze Lv**([czlv24@m.fudan.edu.cn](mailto:czlv24@m.fudan.edu.cn)), and **Wentao Tong**([toldzera@zju.edu.cn](mailto:toldzera@zju.edu.cn)).

Enjoy the code!