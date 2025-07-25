'''
on-device(GPU) energy comsumption monitor for SNN model inference towards samples one-by-one
'''

import os
import time
import torch.utils
from tqdm import tqdm

import torch

from soul.model import *
from soul.neuron import *
from soul.utils import *

# init all config settings
config = init_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# report all configuration
for k, v in sorted(config.items()):
    print(f'{k} = {v}')

print(f'Reproducibility with random seed {config["seed"]}')
init_seed(config["seed"])
print('=' * 50)

print('Load data...')
_, test_dataset, config['input_channels'], config['input_height'], config['input_width'], config['num_classes'] = load_data(
    dataset_dir=config['data_dir'], 
    dataset_type=config['dataset_name'], 
    T=config['time_step']
)

test_loader = test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1, 
    shuffle=False,
    num_workers=config['workers'], 
    pin_memory=True
)

print(f'Load SNN model: {config["model"]} featured {config["neuron_type"].upper()} neuron...')
print(f'surrogate function: {config["surrogate"]}')
config['surrogate_function'] = surrogate_map[config['surrogate']]
config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 

# init model
model = model_map[config['model'].lower()](config)

# load model state dict
best_model_path = os.path.join(config['model_dir'], f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt')
best_params = torch.load(
    best_model_path, 
    map_location='cpu'
)
model.load_state_dict(best_params)
model.to(device)
model.eval()

print('Monitoring energy cost for inference')

printFullReport(getDevice())
pl = PowerLogger(interval=0.05)
pl.start()
time.sleep(5)
pl.recordEvent(name='Process Start')

with torch.inference_mode():
    for inputs, _ in tqdm(test_loader, unit='batch', ncols=80, desc='Inference per sample: '):
        # encoding raw inputs for reasonable SNN operation
        assert len(inputs.shape) in [4, 5], f'Invalid input shape {inputs.shape}...'
        if len(inputs.shape) == 4:
            # (B, C, H, W) -> (T, B, C, H, W)
            inputs = coding_map[config['coding_schema']](inputs, num_steps=config['time_step'])
        else:
            # default event data shape (B, T, C, H, W) -> (T, B, C, H, W)
            inputs = inputs.transpose(0, 1)

        inputs = inputs.to(device)
        model(inputs)  

time.sleep(5)
pl.stop()
total_cost = pl.showDataTraces()
print(str(pl.eventLog))
printFullReport(getDevice())

print(f'energy cost for inference one sample with {config["model"]}: {total_cost / len(test_loader):.2f} J')
