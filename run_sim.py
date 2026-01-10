import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import onnx  # required by compile() export
import torch
from tqdm import tqdm
from soul.backend.neusim import NeuSimArch, compile, convert_spikes
from soul.utils.monitor import BaseMonitor

# Keep the same import style as run_soul.py so model_map/neuron_map/surrogate_map are available.
from soul.model import *   # noqa: F401,F403
from soul.neuron import *  # noqa: F401,F403
from soul.utils import *   # noqa: F401,F403

from collections import defaultdict
import torch

def _targets_to_labels(targets):
    if torch.is_tensor(targets):
        t = targets.detach()
        if t.ndim == 0:
            return [int(t.item())]
        if t.ndim == 1:
            return [int(x) for x in t.tolist()]
        if t.ndim == 2:
            return [int(x) for x in t.argmax(dim=1).tolist()]
        return [int(x) for x in t.view(-1).tolist()]
    else:
        # list / numpy
        try:
            import numpy as np
            t = np.asarray(targets)
            if t.ndim == 0:
                return [int(t)]
            if t.ndim == 1:
                return [int(x) for x in t.tolist()]
            if t.ndim == 2:
                return [int(x) for x in t.argmax(axis=1).tolist()]
            return [int(x) for x in t.reshape(-1).tolist()]
        except Exception:
            return [int(x) for x in targets]

def select_indices_per_class(targets, remaining_per_class):
    labels = _targets_to_labels(targets)
    selected = []
    for i, y in enumerate(labels):
        y = int(y)
        if y in remaining_per_class and remaining_per_class[y] > 0:
            selected.append(i)
            remaining_per_class[y] -= 1
    done = all(v == 0 for v in remaining_per_class.values())
    return selected, done

def main():
    config = init_config()
    log_path = os.path.join(
    config['log_dir'], 
    config['dataset_name'].lower(), 
    config['model'].lower(), 
    config['architecture'].lower()
    )
    ensure_dir(log_path)
    logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])

    config['surrogate_function'] = surrogate_map[config['surrogate']]
    config['neuron'] = neuron_map[config['neuron_type'].lower()](config) 
    
    train_dataset, test_dataset = load_dataset(config)
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle= False,
        num_workers=config['workers'], 
        pin_memory=False,
        drop_last=False,
    )
    model = model_map[config["application"]][config["model"].lower()](config)

    save_model_path = os.path.join(
        config['model_dir'], 
        config['dataset_name'].lower(), 
        config['model'].lower(), 
        config['neuron_type'].lower()
    )

    best_model_path = os.path.join(
        save_model_path, 
        f'best_{config["model"].lower()}_lif_{config["dataset_name"].lower()}_T4_{config["seed"]}.pt'
    )

    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(
            save_model_path, 
            f'best_{config["model"].lower()}_lif_{config["dataset_name"].lower()}_{config["seed"]}.pt'
        )

    best_params = torch.load(
        best_model_path, 
        map_location='cpu'
    )
    device = torch.device("cpu")
    model.load_state_dict(best_params)
    model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    neuron_cls = config["neuron"].__class__
    latency_sum_s = 0.0
    energy_sum_j = 0.0
    n_samples = 0
    latencies = []
    K = 2
    num_classes = int(config["num_classes"])
    total_target = K * num_classes
    pbar = tqdm(total=total_target, desc=f"{config.get('dataset','dataset')} simulate", unit="sample")

    remaining = {c: K for c in range(num_classes)}
    logger.info("Sample partition: ", remaining)
    for bidx, (inputs, targets) in enumerate(loader):
        selected, done = select_indices_per_class(targets, remaining)
        if not selected:
            if done:
                break
            continue
        inputs = inputs.to(device=device, dtype=torch.float32).contiguous().transpose(0, 1)
        input_shape = tuple(inputs.shape[2:])

        if bidx == 0:
            logger.info("Start compilation...")
            arch = NeuSimArch(config['architecture'])
            compile_res = compile(model, input_shape, arch)
            '''
            onnx_path = os.path.join(log_path, "model.onnx")
            json_path = os.path.join(log_path, "neuron_graph.json")

            onnx.save(compile_res.clean_onnx_model, onnx_path)
            Path(json_path).write_text(
                json.dumps(
                    nx.readwrite.json_graph.node_link_data(compile_res.neuron_graph, edges="edges"),
                    indent=2,
                )
            )'''
            logger.info(f"Total neurons: {compile_res.num_neurons}")
            logger.info(f"Total synapses: {compile_res.num_synapses}")
            logger.info(f"Total cores: {compile_res.num_cores}")
            logger.info("=" * 30)

        monitor = BaseMonitor(model, instance=neuron_cls)
        with torch.no_grad():
            _ = model(inputs)
        total_spikes = convert_spikes(compile_res, monitor)  # list/array per sample
        
        for i in selected:
            res = arch.simulate(compile_res, total_spikes[i], packet_size=1, num_threads=16)
            energy_res = arch.estimate_energy(res)

            latency_sum_s += float(res.latency)
            energy_sum_j += float(energy_res.total_energy)
            n_samples += 1

            latencies.append(float(res.latency)) 
            pbar.update(1)
        if n_samples == 0:
            raise RuntimeError("No samples were simulated (empty dataset?).")
        if done:
            break
    pbar.close()

    lat_t = torch.tensor(latencies, dtype=torch.float64)
    std_latency_ms = lat_t.std(unbiased=True).item() * 1e6
    avg_latency_ms = (latency_sum_s / n_samples) * 1e6
    avg_energy_j = (energy_sum_j / n_samples) * 1e6
    
    logger.info(f"Per-class first {K} samples: total_samples={n_samples}")
    
    logger.info(f"Average latency: {avg_latency_ms:.4f} μs")
    logger.info(f"Average energy: {avg_energy_j:.4f} μJ")
    logger.info(f"Latency STD: {std_latency_ms:.4f} μs")
    #energy_res.print_energy_breakdown()

if __name__ == "__main__":
    main()
