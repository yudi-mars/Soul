## TODO List

### 工程任务

- [ ] 2025.08.12-2025.08.30 音频数据处理默认为转换为Mel Spectrogram后喂给模型, 具体模型可能要待定下 @changze
- [ ] 2025.08.15-2025.08.30 Wireless Sensing: 通过无线信号收集的数据(雷达mmWave, wifi signal, RFID这类的)，任务很杂, 我们还是以识别任务为导向 (SenseFi: A Benchmark for WiFi CSI Sensing) 有个开源项目，有数据集和简单的网络结构，很适合我们spike化，可以直接参考 @yudi
    - 4个数据集集成和统一处理接口(从简单到复杂): UT-HAR, NTU-HAR, NTU-HumanID, Widar 3.0
    - 模型结构(选4个就行， 主要是结构多样性):
        - MLP
        - LeNet
        - Spike RNN / LSTM
        - Conv + LSTM
        - ResNet + classifier
        - ViT
- [ ] 2025.08.06-2025.08.10 @changze 加一下IJCAI2025的ILIF(属于membrane potential modulation)，代码和论文都在ILIF.py中（咱们原来的ILIF被改为INTLIF, 'intlif'）
- [ ] 2025.07.10-2025.07.31 @changze @yudi 我们需要ANN2SNN（为vgg，resnet，spikformer至少挑选一个对应可行的资源消耗没那么严重的ANN2SNN训练方法）**从cloud service的视角展示可行性，最后cloud端给边缘端发送模型**
- [ ] 2025.07.20-2025.07.31 @changze @yudi lightweight的方法: ANN->SNN的蒸馏方法~2023cvpr 为vgg，resnet，spikformer提供一个可行的蒸馏方案 **与ANN2SNN同理**
- [ ] 2025.07.20-2025.07.31 @changze lightweight的方法: **NAS**: 如果目前只支持vgg，那就先只支持直通网络,~~但是要确认一些别的方法（比如乐透奖Panda 2022那篇ECCV）看看哪种方法开销更小~~**主要展示的是当前这个NAS方向是如何和我们的Soul融合适配的！**  example(case)导向
- [ ] 2025.07.25-2025.07.31 @changze @yudi quantization接口  **主要展示的是当前这个prune方向是如何和我们的Soul融合适配的**
- [ ] 2025.07.20-2025.08.20 @helin structure/unstructure pruning结果展示
- [ ] TBD Other Sensing Application
    - Multi-Modal Sensing: @changze 是不是用下spike-clip的研究成果稍微展示下？就说这是一种多模态SNN应用？
    - Earable Sensing: 头戴/夹耳式设备之类采集的信号:可以做（1）Sound source localization定位 (2) EEG/EMG生物信号 这两个application SNN的AI模型都有
    - Generative AI for Sensing: 展示一些LLM吐出的token让snn进行解码的示例？@changze
- [ ] TBD @yudi `scripts`里的shell脚本后面应该换成针对utils.app里的各个服务调用的脚本会比较好(run_soul本质就是rate_trainer，后面会放到app/train里)

### 科研任务

- [ ] 2025.08.06-2025.08.31 Softhebb+STDP算法设计@changze | @yudi 负责辅助、代码复审
- [ ] TBD @Lvchangze @Yudi **设计一个针对enrich data的rehearsal-based的增量学习算法（simple but effective）**

--------------------------------------------------------------------------------

# Soul: A Toolbox for Developing Edge Intelligence Applications with Spiking Neural Networks

<p align="center">
    <img src="./assets/code-logo.jpg" width="50%">
</p>

--------------------------------------------------------------------------------

*“I have always been convinced that the only way to get artificial intelligence to work is to do the computation in a way similar to the human brain.”——Geoffrey Hinton*

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Guidance">Guidance</a> •
  <a href="#Results-Analysis">Results Analysis</a> •
  <a href="#Documentation">Documentation</a>
</p>

SOUL (**S**NN-based **O**pen so**U**rce too**L**kit) is developed based on Python and PyTorch for reproducing and developing SNN-based brain-inspired computing algorithms in a unified, comprehensive, and efficient framework for research purposes and practical deployment at the edge.

## Overview

TBD 一张图 @yudi

### Feature

TBD Soul的一些基本的特色

- feature 1
- feature 2

### Dataset Support

1. Vision Sensing
    - [CIFAR10/100](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf) [[Download Link](https://www.cs.toronto.edu/~kriz/cifar.html)]
    - [Tiny-ImageNet](https://ieeexplore.ieee.org/abstract/document/5206848/) [[Download Link](https://www.kaggle.com/c/tiny-imagenet)]
    - [CIFAR10-DVS](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00309/full) [[Download Link](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671)]
    - [DVS-Gesture](https://ieeexplore.ieee.org/document/8100264) [[Download Link](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)]

2. Motion Sensing
    - [UCI HAR](https://www.sciencedirect.com/science/article/abs/pii/S0925231215010930) [[Download Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)]
    - [HHAR](https://dl.acm.org/doi/10.1145/2809695.2809718) [[Download Link](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)]  <!-- *Note: Only Phone-related data is used in this repository* -->
    - [MotionSense](https://dl.acm.org/doi/10.1145/3302505.3310068) [[Download Link](https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset)]
    - [Shoaib](https://www.mdpi.com/1424-8220/14/6/10146) [[Download Link](https://www.researchgate.net/publication/266384007_Sensors_Activity_Recognition_DataSet)]

3. Acoustic Sensing
    <!-- - [GTZAN](https://ieeexplore.ieee.org/abstract/document/1021072) [[Download Link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)] -->
    - [UrbanSound8K](https://dl.acm.org/doi/10.1145/2647868.2655045) [[Download Link](https://urbansounddataset.weebly.com/download-urbansound8k.html)]
    <!-- - [ESC-50](https://dl.acm.org/doi/abs/10.1145/2733373.2806390) [[Download Link](https://github.com/karoldvl/ESC-50/archive/master.zip)] -->
    - [Google Speech Commands](https://arxiv.org/abs/1804.03209) [[Download Link](https://huggingface.co/datasets/google/speech_commands)]
    - [Spiking Heidelberg Digits](https://ieeexplore.ieee.org/document/9311226) [[Download Link](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)]
    - [Spiking Speech Commands](https://ieeexplore.ieee.org/document/9311226) [[Download Link](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)]

4. Wireless Sensing
    - [UT-HAR](https://ieeexplore.ieee.org/document/8067693) [[Download Link](https://github.com/ermongroup/Wifi_Activity_Recognition?tab=readme-ov-file)]
    - [NTU-HAR](https://ieeexplore.ieee.org/document/9667414) [[Download Link](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt)]
    - [NTU-HumanID](https://ieeexplore.ieee.org/abstract/document/9726794) [[Download Link](https://drive.google.com/drive/folders/1R0R8SlVbLI1iUFQCzh_mH90H_4CW2iwt)]
    - [Widar](https://ieeexplore.ieee.org/document/9516988) [[Download Link](https://tns.thss.tsinghua.edu.cn/widar3.0/)]

## Guidance

### Command in Console 

TBD 这个后面等run_soul变成一个app服务后后面再统一做shell脚本修改

- Running `Soul` with a single GPU in default settings
    ```shell
    CUDA_VISIBLE_DEVICES=[GPU-ID] python run_soul.py
    ```

- Running `Soul` with multiple GPUs in default settings
    ```shell
    CUDA_VISIBLE_DEVICES=[GPU-ID1],[GPU-ID2],... torchrun --nproc_per_node=[Number of used GPU] run_soul.py
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

## Results Analysis

TBD 终版前给一个结果展示，比如ESBench的一些结果列上去

## Documentation

TBD (Sphinx Documentation)

## Cite

```
TBD
```

## Contact Us

If there are any questions, please feel free to propose new features by opening an issue or contact with the author: **Di Yu**([yudi2023@zju.edu.cn](mailto:yudi2023@zju.edu.cn)), **Changze Lv**([czlv24@m.fudan.edu.cn](mailto:czlv24@m.fudan.edu.cn)), and **Wentao Tong**([toldzera@zju.edu.cn](mailto:toldzera@zju.edu.cn)).

Enjoy the code!
