import json
from pathlib import Path

import networkx as nx
import numpy as np
import onnx
import torch
import yaml

from soul.backend.neusim import NeuSimArch, convert_spikes
from soul.model.vision import SEWResNet18
from soul.neuron import LIFNode
from soul.utils.monitor import BaseMonitor
from soul.utils.surrogate import surrogate_map

# prepare config and model
lif_conf = yaml.safe_load(open("soul/config/neuron/lif.yaml", "r", encoding="utf-8"))
lif_conf["surrogate_function"] = surrogate_map[lif_conf["surrogate"]]

conf = {
    "num_classes": 10,
    "time_step": 4,
    "input_channels": 1,
    "input_height": 32,
    "input_width": 32,
    "hidden_dim": 1024,
    "neuron": LIFNode(lif_conf),
    "groups": 1,
    "base_width": 64,
    "connect_function": "ADD",
    "width_per_group": 64,
    "zero_init_residual": False,
    "mlp_hidden_dim": 2048,
    "mlp_ratio": 1.0,
}

torch.random.manual_seed(42)
model = SEWResNet18(conf)
model_name = type(model).__name__
input_shape = (1, 32, 32)  # C, H, W

# compile
print("Start compilation...")
arch = NeuSimArch("loihi")
compile_res = arch.compile(model, input_shape)
onnx.save(compile_res.clean_onnx_model, f"{model_name}.onnx")
Path(f"{model_name}.json").write_text(
    json.dumps(
        nx.readwrite.json_graph.node_link_data(compile_res.neuron_graph, edges="edges"),
        indent=2,
    )
)

print(f"Total neurons: {compile_res.num_neurons}")
print(f"Total synapses: {compile_res.num_synapses}")
print(f"Total cores: {compile_res.num_cores}")
print("=" * 30)

# prepare spikes
rng = np.random.default_rng(seed=42)
p = 0.5
batch_size = 2
dummy_input = (
    rng.random(size=(conf["time_step"], batch_size, *input_shape)) < p
).astype(np.uint8)
model.eval()
monitor = BaseMonitor(model, instance=LIFNode)
_ = model(torch.from_numpy(dummy_input).float())
total_spikes = convert_spikes(compile_res, monitor)

# simulate
for i in range(batch_size):
    batch_id = i
    spikes = total_spikes[batch_id]
    assert spikes.shape[0] == compile_res.num_neurons

    true_total_spikes = np.sum(
        np.sum(compile_res.phy_core_conns, axis=1) * np.sum(spikes, axis=1)
    )

    # simulation, using NeuSim
    print("Start simulation...")
    res = arch.simulate(compile_res, spikes, packet_size=1, num_threads=8)

    # print(res.retcode)
    print(f"Total latency: {res.latency * 1e3:.2f} ms")
    print(f"Total cycles: {res.total_cycles}")
    print(f"Total flits: {res.total_recv_flits}")
    print(f"Total/Real spikes: {res.total_firing_cnt}/{np.sum(spikes)}")
    print(f"Total/Real spike packets: {res.total_recv_spikes}/{true_total_spikes}")
    assert res.total_recv_spikes == res.total_sent_spikes == true_total_spikes
    print(f"Total Hops: {res.total_hops}")

    energy_res = arch.estimate_energy(res)
    print(f"Total energy: {energy_res.total_energy} J")
    energy_res.print_energy_breakdown()
