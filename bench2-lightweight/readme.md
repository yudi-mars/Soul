# RQ2: How effectively do basic lightweighting strategies support the deployment of EdgeSNNs?

## Pruning

the argument `thr` with value ranging from 0. to 1. is used to contril the pruning intensity.

For unstructured pruning results targeting at synaptic weight parameters:
```
cd pruning
python main.py -dataset=[dataset name] -data_dir=[your dataset directory] -thr=[Sparsity Ratio] -target=weight 
```

For structured pruning results targeting at filter channels:
```
cd pruning
python main.py -dataset=[dataset name] -data_dir=[your dataset directory] -thr=[Sparsity Ratio] -target=channel
```

The cosole will print the related results according the argument settings.

## Quantization

For quantization-related results, the user can utilize the weight parameters saved in `bench1-neuronal/saved_models` to convert the corresponding quantized weight.

```
python convert.py --dataset=[dataset used] --data_dir=[your dataset directory] --weight_dir=[model weight saved path] --model_type=[model name]
```

After generated the quantized weight, the user can transfer these weights to the edge device and running the evaluation scripts for reusults in RQ2:

```
python test.py --weight_dir=[quantized weight saved directory] --model_type=[model name] --dataset=[dataset used]
```