#!/bin/bash

# 固定参数
GPU_ID=0
BATCH_SIZE=16
DATA_DIR=./data/ncaltech/
DATASET=ncaltech
T=4

# 多个可选参数
seeds=(43)
neurons=(lif)
models=(mlp lenet spikingvgg9 spikingvgg16 sewresnet34 sewresnet50 msresnet34 msresnet50 spikformer256 spikformer384 metaspikeformer256 metaspikeformer384 qkformer256 qkformer384 spikingresformer256 spikingresformer384)

# 循环执行所有组合
for seed in "${seeds[@]}"; do
  for m in "${models[@]}"; do
    echo ">>> Running with seed=${seed}, model=${m}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python run_soul.py \
      -m=$m \
      -b=$BATCH_SIZE \
      -data_dir=$DATA_DIR \
      -dataset=$DATASET \
      -T=$T \
      -seed=$seed

    echo ">>> Done with seed=${seed}, model=${m}"
    echo "---------------------------------------"
  done
done
