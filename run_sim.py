import numpy as np
import yaml

from soul.backend.neusim import NeuSimEnergyModel, compile, sim
from soul.model.vision import SEWResNet50, SpikingMLP
from soul.neuron import LIFNode
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
}
model = SpikingMLP(conf)
# model = SEWResNet18(conf)
# model_name = type(model).__name__
model = SEWResNet50(conf)
input_shape = (3, 32, 32)

print("Start compilation...")
(num_neurons, num_synapses, num_cores), phy_position, phy_core_conns = compile(
    model, input_shape, core_capacity=4096
)

print(f"Total neurons: {num_neurons}")
print(f"Total synapses: {num_synapses}")
print(f"Total cores: {num_cores}")
print("=" * 30)

# prepare spikes
rng = np.random.default_rng(seed=42)
p = 0.5
spikes = (rng.random((num_neurons, conf["time_step"])) < p).astype(np.uint8)

true_total_spikes = np.sum(np.sum(phy_core_conns, axis=1) * np.sum(spikes, axis=1))

# simulation, using NeuSim
print("Start simulation...")
res = sim.run(
    position=phy_position,
    core_conns=phy_core_conns,
    spikes=spikes,
    packet_size=1,
    topology_size=(8, 8),
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
print(res.total_update_cnt, res.total_firing_cnt, num_neurons * conf["time_step"])
print(sop_energy / total_energy)
print(neuron_energy / total_energy)
print(spike_energy / total_energy)
print(hop_energy / total_energy)
print(f"Estimated total energy: {total_energy} J")
