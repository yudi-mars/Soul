from rich.console import Console
from rich.table import Table


class NeuSimEnergyModel:
    def __init__(self, arch_name: str):
        loihi_hop_energy = 4 * 1e-12  # J
        loihi_neuron_energy = 52 * 1e-12  # J
        loihi_spike_energy = 81 * 1e-12  # J
        loihi_sop_energy = 23.6 * 1e-12  # J
        if arch_name.lower() == "loihi":
            self._sop_energy = loihi_sop_energy
        elif arch_name.lower() == "darwin3":
            self._sop_energy = 5.47 * 1e-12  # J
        elif arch_name.lower() == "TrueNorth":
            self._sop_energy = 26 * 1e-12  # J
        elif arch_name.lower() == "spinnaker":
            self._sop_energy = 11.3 * 1e-9  # J
        elif arch_name.lower() == "spinnaker2":
            self._sop_energy = 10 * 1e-12  # J
        else:
            raise NotImplementedError(f"Architecture {arch_name} not supported.")

        self._neuron_energy = loihi_neuron_energy * self._sop_energy / loihi_sop_energy
        self._spike_energy = loihi_spike_energy * self._sop_energy / loihi_sop_energy
        self._hop_energy = loihi_hop_energy * self._sop_energy / loihi_sop_energy

    def estimate_energy(self, sim_res):
        self.sop_energy = self._sop_energy * sim_res.total_recv_spikes
        self.neuron_energy = self._neuron_energy * (
            sim_res.total_update_cnt - sim_res.total_firing_cnt
        )
        self.spike_energy = self._spike_energy * sim_res.total_firing_cnt
        self.hop_energy = self._hop_energy * sim_res.total_hops
        self.total_energy = (
            self.sop_energy + self.neuron_energy + self.spike_energy + self.hop_energy
        )

    def print_energy_breakdown(self):
        console = Console()
        table = Table(title="Energy Breakdown")
        table.add_column("Part", style="cyan")
        table.add_column("Energy (mJ)", style="magenta")
        table.add_column("Pct. (%)", style="green")

        table.add_row(
            "SOP",
            f"{self.sop_energy * 1e3}",
            f"{self.sop_energy / self.total_energy * 100:.2f}",
        )
        table.add_row(
            "Neuorn Update",
            f"{self.neuron_energy * 1e3}",
            f"{self.neuron_energy / self.total_energy * 100:.2f}",
        )
        table.add_row(
            "Neuron Spike",
            f"{self.spike_energy * 1e3}",
            f"{self.spike_energy / self.total_energy * 100:.2f}",
        )
        table.add_row(
            "NoC",
            f"{self.hop_energy * 1e3}",
            f"{self.hop_energy / self.total_energy * 100:.2f}",
        )
        console.print(table)
