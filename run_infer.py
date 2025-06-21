import os
import time
import torch.utils
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=config['workers'], 
    pin_memory=True
)

print(f'Load SNN model: {config["model"]} featured {config["neuron_type"].upper()} neuron...')

model_map = {
    'spikingvgg5': SpikingVGG5, 'spikingvgg9': SpikingVGG9, 'spikingvgg11': SpikingVGG11, 'spikingvgg13': SpikingVGG13, 'spikingvgg16': SpikingVGG16, 'spikingvgg19': SpikingVGG19, 
    'sewresnet18': SEWResNet18, 'sewresnet34': SEWResNet34, 'sewresnet50': SEWResNet50,
    'spikformer2': Spikformer2, 'spikformer4': Spikformer4, 'spikformer8': Spikformer8,
}

neuron_map = {
    "lif": LIFNode,
    "plif": ParametricLIFNode,
    "clif": CLIFNode,
    "glif": GatedLIFNode,
    "ilif": ILIFNode,
    # TODO
}

surrogate_map = {
    'atan': ATan(),
    'erf': Erf(),
    'rect': Rectangular(),
    'sigmoid': FastSigmoid(),
    'quant': Quant(),
    'quant4': Quant4(),
    'rectangle': Rectangle(),
}

print(f'surrogate function: {config["surrogate"]}')
config['surrogate_function'] = surrogate_map[config['surrogate']]
config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 

model = model_map[config['model'].lower()](config)
print('\n'+ str(model))

best_model_path = os.path.join(config['model_dir'], f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt')
best_params = torch.load(
    best_model_path, 
    map_location='cpu'
)
model.load_state_dict(best_params)
model.to(device)

model.eval()
top1_meter = AverageMeter()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        top1_meter.update(acc1.item(), targets.numel())

test_acc = top1_meter.avg
print(f'Test accuracy: {test_acc:.2f}%')

