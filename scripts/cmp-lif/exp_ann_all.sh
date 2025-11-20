#!/bin/bash

# 固定参数
GPU_ID=5
MODEL=lenet
BATCH_SIZE=16

declare -A datasets=(
  # vision sensing datasets
  ["cifar10"]="/home/yudi/data/cifar10/"
  ["cifar100"]="/home/yudi/data/cifar100/"
  ["mnist"]="/home/yudi/data/mnist/"
  ["fashionmnist"]="/home/yudi/data/fashionmnist/"
  ["svhn"]="/home/yudi/data/svhn/"
  # motion sensing datasets
  ["ucihar"]="/home/yudi/data/ucihar/"
  ["hhar"]="/home/yudi/data/hhar/"
  ["motionsense"]="/home/yudi/data/motionsense/"
  ["shoaib"]="/home/yudi/data/shoaib/"
  # acoustic sensing datasets
  ["urbansound"]="/home/yudi/data/urbansound8k/"
  ["gsc"]="/home/yudi/data/gsc/"
  ["gtzan"]="/home/yudi/data/gtzan/"
  ["esc"]="/home/yudi/data/esc50/"
  # wireless sensing
  ["uthar"]="/home/yudi/data/UT_HAR/"
  ["fihar"]="/home/yudi/data/NTU-Fi_HAR/"
  ["fihumanid"]="/home/yudi/data/NTU-Fi-HumanID/"
  ["aril"]="/home/yudi/data/ARIL/"
  # neuromorphic sensing
  ["dvsgesture"]="/home/yudi/data/dvs128gesture/"
  ["cifar10dvs"]="/home/yudi/data/cifar10_dvs/"
  ["ssc"]="/home/yudi/data/ssc/"
  ["shd"]="/home/yudi/data/shd/"
)

# 多个可选参数
seeds=(41 42 43)

# 循环执行所有组合
for dataset in "${!datasets[@]}"; do
  data_dir="${datasets[$dataset]}"

  for seed in "${seeds[@]}"; do
    echo ">>> Running with dataset=${dataset}, seed=${seed}, neuron=ReLU"
    echo ">>> Data directory: ${data_dir}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python run_ann_lenet.py \
      -m=$MODEL \
      -b=$BATCH_SIZE \
      -data_dir=$data_dir \
      -dataset=$dataset \
      -seed=$seed

    echo ">>> Done with dataset=${dataset}, seed=${seed}, neuron=ReLU"
    echo "---------------------------------------"
  done
done
