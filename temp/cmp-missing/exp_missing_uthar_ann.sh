#!/bin/bash

# 固定参数
GPU_ID=0
BATCH_SIZE=16
DATA_DIR=/home/yudi/data/UT_HAR/
DATASET=uthar

# 多个可选参数
phases=(train test)
noises=(0.1 0.2 0.3 0.4)

# 循环执行所有组合
for phase in "${phases[@]}"; do
  for noise in "${noises[@]}"; do
    echo ">>> Running with phase=${phase}, noise=${noise}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python run_ann_lenet.py \
      -b=$BATCH_SIZE \
      -data_dir=$DATA_DIR \
      -dataset=$DATASET \
      -noise=dropout \
      -ni=$noise \
      -phase=$phase

    echo ">>> Done with phase=${phase}, noise=${noise}"
    echo "---------------------------------------"
  done
done