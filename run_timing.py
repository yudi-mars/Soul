"""
Usage:
    python run_stream_latency.py -dataset=mnist -m=lenet -n=lif -T=4 -num_streams 1 -qps 256
"""

import os

import torch

from soul.model import *
from soul.neuron import *
from soul.utils import *

config = init_config()
print('=' * 60)
print(f'Reproducibility with random seed {config["seed"]}')
init_seed(config["seed"])
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset using Soul API
print(f"Loading dataset: {config['dataset_name']}")
_, test_dataset = load_dataset(config)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,  # serial processing one-by-one for latency measurement
    shuffle=False,
    num_workers=0,  # 0 for edge device
    pin_memory=False, # Disable for edge devices
)
num_samples = len(test_loader.dataset)
print(f"Test samples: {num_samples}")

# Load model using Soul API
print(f"Loading model: {config['model']} with {config['neuron_type']} neuron")
config['surrogate_function'] = surrogate_map[config['surrogate']]
config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 

model = model_map[config['application']][config['model'].lower()](config)
best_model_path = os.path.join(
    config['model_dir'], 
    f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_T{config["time_step"]}_{config["seed"]}.pt'
)

best_params = torch.load(
    best_model_path, 
    map_location='cpu',
    weights_only=True
)
model.load_state_dict(best_params)
model.to(device)
model.eval()

# Count parameters
n_params = count_parameters(model, trainable=True)
print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

# Run benchmark 
benchmark = SingleStreamLatency(
    model=model,
    config=config,
    device=device,
    num_queries=min(config['queries_per_stream'] * config['num_streams'], num_samples),
    warmup_runs=config['warmup_runs'],
)
result = benchmark.run_benchmark(test_loader)

print("\n" + "=" * 60)
print("SINGLE-STREAM RESULTS")
print("=" * 60)
for k, v in result.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")
print("=" * 60 + "\n")
