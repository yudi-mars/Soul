import numpy as np

from .energy import NeuSimEnergyModel


class NeuSimArch:
    def __init__(self, arch_name: str):
        self.arch_name = arch_name.lower()
        if self.arch_name == "loihi":
            self.neurons_per_core = 1024
            self.frequecy_mhz = 100  # 10ns per neuron update
        elif self.arch_name == "darwin3":
            self.neurons_per_core = 4096
            self.frequecy_mhz = 400
        elif self.arch_name == "truenorth":
            self.neurons_per_core = 256
            self.frequecy = 1e3  # fixed 1KHz timestep
        elif self.arch_name == "spinnaker":
            self.neurons_per_core = 1000
            self.frequecy_mhz = 200
        elif self.arch_name == "spinnaker2":
            self.neurons_per_core = 1000
            self.frequecy_mhz = 400
        else:
            raise NotImplementedError(f"Architecture {arch_name} not supported.")
        self.energy_model = NeuSimEnergyModel(arch_name)

    def compile(self, model, input_shape: tuple[int, ...]):
        from .compile import compile

        return compile(model, input_shape, self)

    def simulate(
        self,
        compile_res,
        spikes,
        packet_size: int = 1,
        num_threads: int = 1,
    ):
        from . import sim

        meshy = int(np.sqrt(compile_res.num_cores))
        meshx = (compile_res.num_cores // meshy) + (
            1 if compile_res.num_cores % meshy != 0 else 0
        )

        res = sim.run(
            position=compile_res.phy_position,
            core_conns=compile_res.phy_core_conns,
            spikes=spikes,
            packet_size=packet_size,
            topology_size=(meshy, meshx),
            num_threads=num_threads,
        )
        if self.arch_name == "truenorth":
            res.latency = res.num_timesteps * (1 / self.frequecy)
        else:
            res.latency = res.total_cycles * (1e-6 / self.frequecy_mhz)

        return res

    def estimate_energy(self, sim_res):
        return self.energy_model.estimate_energy(sim_res)
