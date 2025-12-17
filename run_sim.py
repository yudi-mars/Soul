import json
from pathlib import Path

import networkx as nx
import numpy as np
import onnx
import torch
import yaml

from soul.backend.neusim import NeuSimEnergyModel, compile, convert_spikes, sim
from soul.model.vision import SEWResNet18, SpikingMLP
from soul.neuron import LIFNode
from soul.utils.monitor import BaseMonitor
from soul.utils.surrogate import surrogate_map

lif_conf = yaml.safe_load(open("soul/config/neuron/lif.yaml", "r", encoding="utf-8"))
lif_conf["surrogate_function"] = surrogate_map[lif_conf["surrogate"]]

conf = {
    "num_classes": 10,
    "time_step": 4,
    "input_channels": 3,
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

model = SpikingMLP(conf)
model = SEWResNet18(conf)
model_name = type(model).__name__
# model = SpikingVGG9(conf)
input_shape = (3, 32, 32)

print("Start compilation...")
compile_res = compile(model, input_shape, core_capacity=4096)
onnx.save(compile_res.clean_onnx_model, f"{model_name}.onnx")
Path(f"{model_name}").write_text(
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
dummy_input = (rng.random(size=(conf["time_step"], 1, *input_shape)) < p).astype(
    np.uint8
)
model.eval()
monitor = BaseMonitor(model, instance=LIFNode)
_ = model(torch.from_numpy(dummy_input).float())
spikes = convert_spikes(compile_res, monitor)
batch_id = 0
spikes = spikes[batch_id]
assert spikes.shape[0] == compile_res.num_neurons

true_total_spikes = np.sum(
    np.sum(compile_res.phy_core_conns, axis=1) * np.sum(spikes, axis=1)
)

# simulation, using NeuSim
print("Start simulation...")
meshy = int(np.sqrt(compile_res.num_cores))
meshx = (compile_res.num_cores // meshy) + (
    1 if compile_res.num_cores % meshy != 0 else 0
)

res = sim.run(
    position=compile_res.phy_position,
    core_conns=compile_res.phy_core_conns,
    spikes=spikes,
    packet_size=1,
    topology_size=(meshy, meshx),
    num_threads=8,
)
# print(res.retcode)
print(f"Total cycles: {res.total_cycles}")
print(f"Total flits: {res.total_recv_flits}")
print(f"Total/Real spikes: {res.total_firing_cnt}/{np.sum(spikes)}")
print(f"Total/Real spike packets: {res.total_recv_spikes}/{true_total_spikes}")
assert res.total_recv_spikes == res.total_sent_spikes == true_total_spikes
print(f"Total Hops: {res.total_hops}")

energy_model = NeuSimEnergyModel("loihi")
total_energy, (sop_energy, neuron_energy, spike_energy, hop_energy) = (
    energy_model.estimate_energy(res, detail=True)
)
# print(
#     res.total_update_cnt,
#     res.total_firing_cnt,
#     compile_res.num_neurons * conf["time_step"],
# )
# print(sop_energy / total_energy)
# print(neuron_energy / total_energy)
# print(spike_energy / total_energy)
# print(hop_energy / total_energy)
print(f"Total energy: {total_energy} J")
