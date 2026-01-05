'''
Count OPs and estimate theoretical energy cost for SNNs and ANNs
for ANNs, the OPs refer to FLOPs (Floating Point Operations)
for SNNs, the OPs refer to SOPs (Synapse Operations) and FLOPs for the first layer
'''

import os
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
_, test_dataset = load_dataset(config)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=config['workers'], 
    pin_memory=True
)

config['surrogate_function'] = surrogate_map[config['surrogate']]
config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 

# init model
model = model_map[config['application']][config['model'].lower()](config)

# load model state dict
best_model_path = os.path.join(
    config['model_dir'], 
    f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_T{config["time_step"]}_{config["seed"]}.pt'
)
best_params = torch.load(
    best_model_path, 
    map_location='cpu'
)
model.load_state_dict(best_params)
model.to(device)
model.eval()

# calculate theoretical energy cost per sample inference
print('Counting FLOPs/SOPs for theoretical inference cost')
ops_monitor(model, is_sop=config['sop'])
for inputs, _ in tqdm(test_loader, unit='batch', ncols=80, desc='Count OPs: '):
    inputs = inputs.to(device, non_blocking=True)
    # default data shape (B, T, input_size) -> (T, B, input_size)
    inputs = inputs.transpose(0, 1)
    _ = model(inputs)


# calculate average energy cost per sample inference
total_sops = 0
total_flops = 0
for k, v in MODULE_SOP_DICT.items():
    total_sops += v
for k,v in MODULE_FLOPS_DICT.items():
    total_flops += v
avg_sops = total_sops / len(test_loader)
avg_flops = total_flops / len(test_loader)
avg_ops = avg_sops + avg_flops
avg_energy_per_sample = avg_sops * config['e_ac'] + avg_flops * config['e_mac']
print(f"Average number of SOPs for model {config['model']} inference per sample: {avg_sops / 1e6:.2f} M")
print(f"Average number of FLOPs for model {config['model']} inference per sample: {avg_flops / 1e6:.2f} M")
print(f"Average number of Operations (#OPs): {avg_ops:.2f}, corresponding theoretical energy cost: {avg_energy_per_sample / 1e9:.2f} mJ")
