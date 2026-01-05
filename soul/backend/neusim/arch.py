from .energy import NeuSimEnergyModel


class NeuSimArch:
    def __init__(self, arch_name: str):
        if arch_name.lower() == "loihi":
            self.neurons_per_core = 1024
        elif arch_name.lower() == "darwin3":
            self.neurons_per_core = 4096
        elif arch_name.lower() == "truenorth":
            self.neurons_per_core = 256
        elif arch_name.lower() == "spinnaker":
            self.neurons_per_core = 1000
        elif arch_name.lower() == "spinnaker2":
            self.neurons_per_core = 1000
        else:
            raise NotImplementedError(f"Architecture {arch_name} not supported.")
        self.energy_model = NeuSimEnergyModel(arch_name)

    def estimate_energy(self, sim_res):
        return self.energy_model.estimate_energy(sim_res)
