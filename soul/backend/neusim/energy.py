class NeuSimEnergyModel:
    def __init__(self, arch_name: str):
        loihi_hop_energy = 4 * 1e-12  # J
        loihi_neuron_energy = 52 * 1e-12  # J
        loihi_spike_energy = 81 * 1e-12  # J
        loihi_sop_energy = 23.6 * 1e-12  # J
        if arch_name.lower() == "loihi":
            self.sop_energy = loihi_sop_energy
        elif arch_name.lower() == "darwin3":
            self.sop_energy = 5.47 * 1e-12  # J
        elif arch_name.lower() == "TrueNorth":
            self.sop_energy = 26 * 1e-12  # J
        elif arch_name.lower() == "spinnaker":
            self.sop_energy = 11.3 * 1e-9  # J
        elif arch_name.lower() == "spinnaker2":
            self.sop_energy = 10 * 1e-12  # J
        else:
            raise NotImplementedError(f"Architecture {arch_name} not supported.")

        self.neuron_energy = loihi_neuron_energy * self.sop_energy / loihi_sop_energy
        self.spike_energy = loihi_spike_energy * self.sop_energy / loihi_sop_energy
        self.hop_energy = loihi_hop_energy * self.sop_energy / loihi_sop_energy

    def estimate_energy(self, sim_res, detail=False) -> float:
        sop_energy = self.sop_energy * sim_res.total_recv_spikes
        neuron_energy = self.neuron_energy * (
            sim_res.total_update_cnt - sim_res.total_firing_cnt
        )
        spike_energy = self.spike_energy * sim_res.total_firing_cnt
        hop_energy = self.hop_energy * sim_res.total_hops
        total_energy = sop_energy + neuron_energy + spike_energy + hop_energy
        if detail:
            return total_energy, (sop_energy, neuron_energy, spike_energy, hop_energy)
        return total_energy
