# Soul

SNN-based open source toolkit for foundation model benchmarking.

Coming Soon...

## Overview

TBD

## Requirements

```
TBD
```

## How to run
### Command in Console 
- Running `Soul` with single GPU in default settings
    ```
    CUDA_VISIBLE_DEVICES=[GPU ID] python run_soul.py
    ```

- Running `Soul` with multiple GPUs in default settings
    ```
    CUDA_VISIBLE_DEVICES=[GPU ID1],[GPU ID2],... torchrun --nproc_per_node=[Number of used GPU] run_soul.py
    ```

### Documentation

TBD


## TODO List

- [ ] TBD @wentao monitor加上对nn.linear的统计
- [ ] 2025.06.23-2025.12.31 @Lvchangze **设计一个适合edge的STDP few-shot learning方法，input需要满足sequential的特点(即不是repeat input, 而是按time step切分图片或序列, 以一种先后顺序的输入完成learning)**
- [ ] 2025.06.17-2025.06.25 @changze Firing Mechanism类目下的神经元建模，能融合的都要融合进来
    - RSN[83]未开源
    - GLIF[84]已完成
    - ~~ALIF[85]未开源~~
    - PSN[86]已完成
    - TLIF[87]已完成
    - ~~DCIS[88]这篇不是神经元，而是一个特殊的卷积结构 (已在survey删除)~~
    - T-RevSNN[89]官方代码中的神经元实现和I-LIF完全相同，其实就是整数脉冲，都是man yao做的
    - I-LIF[90]已实现
    - IE-LIF[91]已实现
    - ~~P-SpikeSSM [92]逆天印度人，他们整个spikessm网络的代码就不是snn，没有spiking neuron的概念~~
    - ST-BIF @Changze **记得看下这个cvpr2025的LIF**
- [ ] 2025.06.23-2025.07.02 @changze Potential Modulation类目下的神经元建模，能融合的都要融合进来
- [ ] 2025.06.23-2025.06.24 @Changze 维护survey第一版的table 1
- [x] 2025.06.17-2025.06.24 @Changze @yudi 大survey中的table 1 Survey Comparison列举的survey精读，然后最后一天讨论
- [ ] 2025.06.17-2025.07.02 @yudi Toplogy实现：~~MS-ResNet~~, Spike-driven Transformer V2 (meta-spikformer), SpikingResFormer, QKFormer
- [x] 2025.06.17-2025.06.19 Survey hybrid learning章节补充
- [x] 2025.06.19-2025.06.22 Survey 重新整理hardware software部分，并与future direction合并扩写
- [ ] TBD @yudi run_soul独立训练、推理、算各类指标的接口

## Cite

```
TBD
```
