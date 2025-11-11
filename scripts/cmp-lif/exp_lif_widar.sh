#!/bin/bash

# 固定参数
GPU_ID=5
MODEL=lenet
BATCH_SIZE=16
DATA_DIR=/home/yudi/data/Widardata/
DATASET=widar
T=4

# 多个可选参数
seeds=(41 42 43)
neurons=(clif glif intlif lif plif psn tlif ilif rplif)

# 循环执行所有组合
for seed in "${seeds[@]}"; do
  for n in "${neurons[@]}"; do
    echo ">>> Running with seed=${seed}, neuron=${n}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python run_soul.py \
      -m=$MODEL \
      -b=$BATCH_SIZE \
      -data_dir=$DATA_DIR \
      -dataset=$DATASET \
      -T=$T \
      -n=$n \
      -seed=$seed

    echo ">>> Done with seed=${seed}, neuron=${n}"
    echo "---------------------------------------"
  done
done
