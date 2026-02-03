# NeuSim backend

## Build from source

```shell
git submodule sync --init --recursive
cmake -B build
cmake --build build
```

## Usage
There is an example in `run_sim.py`.

1. prepare SNN model
```python
from soul.model.vision import SEWResNet18
from soul.neuron import LIFNode

model = SEWResNet18(conf)
input_shape = (3, 32, 32)
```

2. compile to NeuSim backend
```python
from soul.backend.neusim import NeuSimArch

arch = NeuSimArch("loihi")
compile_res = arch.compile(model, input_shape)
```

3. prepare spikes
```python
from soul.neuron import LIFNode
from soul.utils.monitor import BaseMonitor
from soul.backend.neusim import convert_spikes

dummy_input = torch.rand(conf["time_step"], 1, *input_shape)
model.eval()
monitor = BaseMonitor(model, instance=LIFNode)
_ = model(torch.from_numpy(dummy_input).float())
spikes = convert_spikes(compile_res, monitor) # shape: (batch, num_neurons, timestep)
batch_id = 0
spikes = spikes[batch_id] # shape: (num_neurons, timestep)
```


4. run NeuSim
```python
from soul.backend.neusim import sim

meshy = int(np.sqrt(compile_res.num_cores))
meshx = (compile_res.num_cores // meshy) + (1 if compile_res.num_cores % meshy != 0 else 0)

res = arch.simulate(compile_res, spikes, packet_size=1, num_threads=8)
```

5. print result
```python
from soul.backend.neusim import NeuSimEnergyModel

print(f"Total latency: {res.latency * 1e3:.2f} ms")
print(f"Total cycles: {res.total_cycles}")
print(f"Total flits: {res.total_recv_flits}")
print(f"Total spikes: {res.total_firing_cnt}")
print(f"Total spike packets: {res.total_recv_spikes}")
print(f"Total Hops: {res.total_hops}")

energy_res = arch.estimate_energy(res)
print(f"Total energy: {energy_res.total_energy} J")
energy_res.print_energy_breakdown()
```
