<p align="center">
    <img src="./assets/code-logo.jpg" width="50%">
</p>

--------------------------------------------------------------------------------

# Soul

*“I have always been convinced that the only way to get artificial intelligence to work is to do the computation in a way similar to the human brain.”——Geoffrey Hinton*


SOUL (**S**NN-based **O**pen so**U**rce too**L**kit) is developed based on Python and PyTorch for reproducing and developing SNN-based brain-inspired computing algorithms in a unified, comprehensive, and efficient framework for research purposes and practical deployment at the edge.

## Overview

TBD

## Feature

TBD 

- feature 1
- feature 2

## Requirements

```
TBD
```

## How to run
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

## Documentation

TBD 后面学生来做这个部分的事情


## TODO List

随着更多模态数据的引入, Soul执行需要额外的参数来对不同应用场景进行区分, 此外各类训练方式的引入也会对Soul的原本设计造成影响，架构可能要再变动下 @yudi，要随时沟通！

### 工程任务

- [x] 2025.07.03-2025.07.05 @yudi run_soul独立训练、推理、算各类指标的接口,提供一些单一功能的执行文件
- [x] ~~2025.07.05-TBD @wentao 目前count_sops还无法对transformer相关模型进行统计(或者统计很慢), 需要优化调整~~
- [ ] 2025.07.02-2025.07.15 Speech Recognition相关文献调研、模型以及数据加载复现以及融入Soul的难点评估 @changze
- [ ] 2025.07.10-2025.07.31 @changze 我们需要ANN2SNN（为vgg，resnet，spikformer至少挑选一个对应可行的资源消耗没那么严重的ANN2SNN训练方法）
- [ ] 2025.07.20-2025.07.31 @changze lightweight的方法: ANN->SNN的蒸馏方法~2023cvpr 为vgg，resnet，spikformer提供一个可行的蒸馏方案 
- [ ] 2025.07.20-2025.07.31 @changze lightweight的方法: NAS: 如果目前只支持vgg，那就先只支持直通网络，但是要确认一些别的方法（比如乐透奖Panda 2022那篇ECCV）看看哪种方法开销更小 
- [ ] 2025.07.25-2025.07.31 @changze quantization接口 
- [ ] 2025.07.11-2025.07.31 @yudi @helin prune： structure pruning支持GPU/neuromorphic， unstructure pruning支持neuromorphic （目前提供GPU模拟即可）
- [ ] 2025.07.02-2025.07.15 Human Activity Recognition相关模型，数据加载模块融入Soul @yudi
- [ ] 2025.06.23-2025.07.02 @changze Potential Modulation类目下的神经元建模，能融合的都要融合进来
    - PLIF[70] 已经完成
    - ~~SRIF[71] guoyufei的，发现无法“即插即用”，需要指定input dim和output dim~~
    - LTMD [72] 已实现
    - ~~BDETT [73] 未开源~~
    - ~~DTA-TTFS [74] 未开源~~
    - CLIF[75] 已经完成
    - ~~ABN [76] 未开源~~
    - ~~DA-LIF [77] 未开源~~
    - ~~Smooth LIF [78] 未开源~~

### 科研任务
- [ ] 2025.07.01-2025.07.31 @Lvchangze @Yudi **设计一个针对enrich data的rehearsal-based的增量学习算法（simple but effective）**
- [ ] 2025.07.01-2025.08.31 @changze STDP训练方法调研 提供一个好用的方案，目前可以给个GPU方案，但需要给出neuromorphic chip的部署基本逻辑
- [ ] 2025.07.01-2025.08.31 @yudi  temporal learning 提供一个好用的方案，可以GPU方针，但需要给出neuromorphic chip的部署基本逻辑


## Cite

```
TBD
```
