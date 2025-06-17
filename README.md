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

- [ ] 时间待定 @wentao monitor加上对nn.linear的统计
- [ ] 2025.06.17-2025.06.25 Firing Mechanism
- [ ] 2025.06.23-2025.07.02 Potential LIF
- [ ] 2025.06.17-2025.06.24 Survey Comparison以及讨论
- [ ] 2025.06.17-2025.07.02 Toplogy实现：MS-ResNet, Spikformer V2, SpikingResFormer, QKFormer

## Cite

```
TBD
```
