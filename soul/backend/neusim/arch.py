import numpy as np

from .energy import NeuSimEnergyModel


class NeuSimArch:
    def __init__(self, arch_name: str):
        self.arch_name = arch_name.lower()
        if self.arch_name == "loihi":
            self.neurons_per_core = 1024
            self.frequecy_mhz = 100  # 10ns per neuron update
            self.sram_per_core_KB = 208
            self.weight_precision_bits = 8
        elif self.arch_name == "darwin3":
            self.neurons_per_core = 4096
            self.frequecy_mhz = 400
            self.sram_per_core_KB = 768
            self.weight_precision_bits = 8
        elif self.arch_name == "truenorth":
            self.neurons_per_core = 256
            self.frequecy = 1e3  # fixed 1KHz timestep
            self.sram_per_core_KB = 12.75
            self.weight_precision_bits = 8
        elif self.arch_name == "spinnaker":
            self.neurons_per_core = 1000
            self.frequecy_mhz = 200
            self.sram_per_core_KB = (
                96 + 128 * 1024 / 20
            )  # 96KB SRAM + 128MB shared SDRAM
            self.weight_precision_bits = 8
        elif self.arch_name == "spinnaker2":
            self.neurons_per_core = 1000
            self.frequecy_mhz = 400
            self.sram_per_core_KB = 128
            self.weight_precision_bits = 8
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
        try:
            from . import sim
        except ImportError:
            raise ImportError(
                "NeuSim backend is not properly installed. Please follow the installation guide in soul/backend/neusim/README.md"
            )

        res = sim.run(
            position=compile_res.phy_position,
            core_conns=compile_res.phy_core_conns,
            spikes=spikes,
            packet_size=packet_size,
            topology_size=compile_res.topology_size,
            num_threads=num_threads,
        )
        if self.arch_name == "truenorth":
            res.latency = res.num_timesteps * (1 / self.frequecy)
        else:
            res.latency = res.total_cycles * (1e-6 / self.frequecy_mhz)

        res.memory_usage = (
            compile_res.num_cores * self.sram_per_core_KB / 1024
            + compile_res.num_params * self.weight_precision_bits / 8 / 1024 / 1024
        )  # in MB

        return res

    def estimate_energy(self, sim_res):
        return self.energy_model.estimate_energy(sim_res)
