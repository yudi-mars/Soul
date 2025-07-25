# RQ1: How do neuronal behaviors and architectures impact the deployability of EdgeSNNs?

In the main text, we have discussed that the components of SNNs can be divided into two compartments: **spiking neurons** and **network topologies**. 

For spiking neurons, we choose the commonly-used `LIF` as a simple baseline and some LIF variants by modulating the firing mechanism (`ILIF`, `GLIF`) and the potential dynamics (`PLIF`, `CLIF`), whose implementations are all based on `SpikingJelly` and are available in the `model` folder.

For network topologies, we choose three representative SNN networks for CV tasks, i.e., VGG-structured, ResNet-structured, and Transformer-structured.

## How to run:

### Training:

For training stage, we need to use the `train.py` to prepare the model parameter weight files for across all cases.

```
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py --T 4 --num_classes 10 --epochs 100 --batch_size 64 --dataset CIFAR10 --data_dir [dataset path] --lr 5e-4 --neuron_type [LIF model] --model_type [model name] > [logfile name].log
```

The generated weight will be transfered to each edge device for inference-related evaluation

### Testing in Jetson Orin AGX

After moving related model parameter file to your desired path, you can run the following command to start evaluation.

```
python test.py -dataset=[dataset_name] -seed=[seed_num] -neuron_type=[neuron_type]
```

_Note: Jetson devices introduce embedded power sensors to measure the actual running power for the devices, which can be used to calculate the practical energy consumption for on-device SNNs when implementing inference tasks. More details are included in the Powerlogger class in `power_check.py`._

### Shell scripts for all model preparation

please switch to the `shell` folder, we have provided some useful shell commands that can directly present all related results reported in our paper.

## Future Work

More LIF variants and SNN topologies are included in the next version of ESBench. For current benchmark, we just use those representative models to demonstrate some intriguing findings.