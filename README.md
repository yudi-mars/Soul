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
- [ ] 2025.06.17-2025.06.25 @changze Firing Mechanism类目下的神经元建模，能融合的都要融合进来
- [ ] 2025.06.23-2025.07.02 @changze Potential Modulation类目下的神经元建模，能融合的都要融合进来
- [ ] 2025.06.17-2025.06.24 @Changze @yudi 大survey中的table 1 Survey Comparison列举的survey精读，然后最后一天讨论
- [ ] 2025.06.17-2025.07.02 @yudi Toplogy实现：MS-ResNet, Spikformer V2, SpikingResFormer, QKFormer
- [x] 2025.06.17-2025.06.19 Survey hybrid learning章节补充
- [ ] 2025.06.19-2025.06.22 Survey 重新整理hardware software部分，并与future direction合并扩写
- [ ] TBD @yudi run_soul独立训练、推理、算各类指标的接口

## Cite

```
TBD
```
