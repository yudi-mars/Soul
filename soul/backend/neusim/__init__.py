from .compile import compile, convert_spikes
from .energy import NeuSimEnergyModel

try:
    from . import sim
except ImportError:
    raise ImportError(
        "NeuSim backend is not properly installed. Please follow the installation guide in soul/backend/neusim/README.md"
    )

__all__ = ["compile", "convert_spikes", "NeuSimEnergyModel", "sim"]
