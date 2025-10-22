#!/bin/bash

# Not Recommend to run this script to compare all LIF models on ImageNet, since LeNet structure is too simple.

# 固定参数
GPU_ID="3,4"
MODEL=lenet
BATCH_SIZE=16
DATA_DIR=/home/yudi/data/tiny-imagenet-200/
DATASET=imagenet
T=4

# 多个可选参数
seeds=(41 42 43)
neurons=(clif glif ielif intlif lif ltmd plif psn tlif ilif)


# 循环执行所有组合
for seed in "${seeds[@]}"; do
  for n in "${neurons[@]}"; do
    echo ">>> Running with seed=${seed}, neuron=${n}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    torchrun --nproc_per_node=2 run_soul.py \
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

  