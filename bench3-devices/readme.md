# RQ3: How does device heterogeneity influence the performance of deployed EdgeSNNs?

## Prepare the model

```
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 4 -b 64 -dataset CIFAR10 -data_dir [dataset path] > ./logs/[logfile name].log
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 4 -b 16 -dataset TinyImageNet -data_dir [dataset path] > ./logs/[logfile name].log
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 10 -b 4 -dataset CIFAR10DVS -data_dir [dataset path] > ./logs/[logfile name].log
CUDA_VISIBLE_DEVICES=[GPU ID] python train.py -T 16 -b 16 -dataset DVSGesture -data_dir [dataset path] > ./logs/[logfile name].log
```

## test the latency at the edge

After transfering the trained weight parameter files to the corresponding device platform, implement the following command to get the results of RQ3 (file path should be customized according to the user preference).

```
python test.py
```

_Note: For the Edge2Server results_

We first need to run the flask framework with a preloaded model at the server first:
```
python service.py
```
This command will activate the flask server and awaiting for the device to send model input to the server-side SNN model.

Then running `local.py` on devices with command:
```
python local.py
```

The console at edge device will provide the final results.