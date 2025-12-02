import numpy as np
import torch
import tqdm
import yaml

from soul import sim
from soul.model.vision import SpikingMLP
from soul.model.vision.general import multi_time_forward
from soul.neuron import LIFNode, functional
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
}
model = SpikingMLP(conf)


def dump_SpikingMLP(model, x):
    functional.reset_net(model)
    neurons = []
    synapses = []

    x = x.flatten(2)  # (T, B, C, H, W) -> (T, B, CHW)

    x = multi_time_forward(x, model.ln1)
    x = model.lif1(x)
    neurons.append(x.shape[2:])
    synapses.append(
        {
            "from": -1,
            "to": 0,
            "layers": [("flatten",), ("linear",)],
        }
    )

    x = multi_time_forward(x, model.ln2)
    x = model.lif2(x)
    neurons.append(x.shape[2:])
    synapses.append(
        {
            "from": 0,
            "to": 1,
            "layers": [("linear",)],
        }
    )

    x = model.head(x.mean(0))  # (T, B, D) -> (B, D)

    return x, (None), neurons, synapses


x = torch.zeros(1, 1, 3, 32, 32)  # (T, B, C, H, W)
_, _, neurons, synapses = dump_SpikingMLP(model, x)


def dump_linear(in_dim, out_dim):
    num_conns = in_dim * out_dim
    src = np.arange(in_dim, dtype=np.int32)
    src = src[np.newaxis, :]
    src = np.repeat(src, out_dim, axis=0)
    return src, np.arange(out_dim, dtype=np.int32), num_conns


def dump_layers(in_shape, out_shape, layers):
    synapses = None
    for layer in layers:
        layer_type = layer[0]
        if layer_type == "linear":
            synapses = dump_linear(in_shape[-1], out_shape[-1])
    return synapses


for syn in synapses:
    if syn["from"] == -1:
        continue
    in_shape = neurons[syn["from"]]
    out_shape = neurons[syn["to"]]
    syn["synapses"] = dump_layers(in_shape, out_shape, syn["layers"])

num_neurons_layer = [np.prod(neu) for neu in neurons]
num_neurons = np.sum(num_neurons_layer)
neuron_start = np.cumsum([0] + num_neurons_layer[:-1])
print(f"Total neurons: {num_neurons}")
print(
    f"Total synapses: {np.sum([syn['synapses'][-1] for syn in synapses if 'synapses' in syn])}"
)

# partition
core_capacity = 64
num_cores = int(np.ceil(num_neurons / core_capacity))
mesh_y = int(np.ceil(np.sqrt(num_cores)))
mesh_x = mesh_y
print(f"Num cores: {num_cores}, {mesh_y}x{mesh_x}")
num_neurons_core = np.ceil(num_neurons / num_cores).astype(np.int32)
position = np.arange(num_neurons) // num_neurons_core


def dump_core_conns(src, dst, pre_num, post_idx, pos, num_cores, use_tqdm=False):
    core_conns = np.zeros([pre_num, num_cores], dtype=np.bool_)
    if use_tqdm:
        pbar = tqdm.tqdm(total=len(dst))
    for ss, d in zip(src, dst):
        d_core = pos[post_idx + d]
        core_conns[ss, d_core] = True
        if use_tqdm:
            pbar.update(1)
    if use_tqdm:
        pbar.close()
    return core_conns


core_conns = np.zeros([num_neurons, num_cores], dtype=np.uint8)
for syn in synapses:
    if "synapses" not in syn:
        continue
    src, dst, _ = syn["synapses"]
    pre_num = num_neurons_layer[syn["from"]]
    pre_idx = neuron_start[syn["from"]]
    post_idx = neuron_start[syn["to"]]
    core_conns[pre_idx : pre_idx + pre_num] = dump_core_conns(
        src,
        dst,
        pre_num,
        post_idx,
        position,
        num_cores,
        use_tqdm=True,
    )

# mapping
mapping_l2p = np.arange(num_cores)
mapping_p2l = np.arange(num_cores)
for i in range(num_cores):
    mapping_p2l[mapping_l2p[i]] = i
phy_position = mapping_l2p[position].astype(np.uint32)
phy_core_conns = core_conns[:, mapping_p2l].astype(np.uint8)

# prepare spikes
rng = np.random.default_rng(seed=42)
p = 0.1
spikes = (rng.random((num_neurons, conf["time_step"])) < p).astype(np.uint8)

true_total_spikes = np.sum(np.sum(phy_core_conns, axis=1) * np.sum(spikes, axis=1))

# simulation, using NeuSim
res = sim.run(
    position=phy_position,
    core_conns=phy_core_conns,
    spikes=spikes,
    packet_size=1,
    topology_size=(mesh_y, mesh_x),
    num_threads=1,
)
# print(res.retcode)
print(res.total_cycles)
print(res.total_recv_flits)
print(res.total_firing_cnt, np.sum(spikes))
print(res.total_recv_spikes, res.total_sent_spikes, true_total_spikes)
assert res.total_recv_spikes == true_total_spikes
