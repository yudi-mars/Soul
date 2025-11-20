#!/bin/bash

# 固定参数
GPU_ID=3
BATCH_SIZE=16
DATA_DIR=/home/yudi/data/urbansound8k/
DATASET=urbansound

# 多个可选参数
phases=(train test)
noises=(0 0.1 0.25 0.5 1)

# 循环执行所有组合
for phase in "${phases[@]}"; do
  for noise in "${noises[@]}"; do
    echo ">>> Running with phase=${phase}, noise=${noise}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python run_ann_lenet.py \
      -b=$BATCH_SIZE \
      -data_dir=$DATA_DIR \
      -dataset=$DATASET \
      -noise=gaussian \
      -ni=$noise \
      -phase=$phase

    echo ">>> Done with phase=${phase}, noise=${noise}"
    echo "---------------------------------------"
  done
done