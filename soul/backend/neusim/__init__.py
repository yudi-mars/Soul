from .arch import NeuSimArch
from .compile import compile, convert_spikes

try:
    from . import sim
except ImportError:
    raise ImportError(
        "NeuSim backend is not properly installed. Please follow the installation guide in soul/backend/neusim/README.md"
    )

__all__ = ["compile", "convert_spikes", "NeuSimArch", "sim"]
