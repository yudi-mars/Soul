"""
Spiking neurons of Soul. Every neuron is built from the global config dict: ``neuron_map[name](config)``.

    MemoryModule                      stateful module, memories restored by reset()
    ├── BaseNode                      charge -> fire -> reset template
    │   ├── LIFNode
    │   │   ├── CLIFNode
    │   │   ├── ILIFNeuron
    │   │   └── RPLIFNode
    │   ├── ParametricLIFNode
    │   ├── INTLIFNode
    │   │   └── TLIFNode
    │   ├── STBIF
    │   └── AdaptBaseNode
    │       └── IzhikevichNode
    └── GatedLIFNode                  multi-step only
    nn.Module (stateless, whole sequence at once)
    ├── ParallelSpikingNode
    ├── IELIFNode
    └── LTMD
"""
from .base import StepModule, MemoryModule, BaseNode, AdaptBaseNode
from .LIF import LIFNode
from .CLIF import CLIFNode
from .ILIF import ILIFNeuron
from .RPLIF import RPLIFNode
from .PLIF import ParametricLIFNode
from .INTLIF import INTLIFNode
from .TLIF import TLIFNode
from .STBIF import STBIF
from .Izhikevich import IzhikevichNode
from .GLIF import GatedLIFNode
from .PSN import ParallelSpikingNode
from .IELIF import IELIFNode
from .LTMD import LTMD

neuron_map = {
    "lif": LIFNode,
    "plif": ParametricLIFNode,
    "clif": CLIFNode,
    "glif": GatedLIFNode,
    "intlif": INTLIFNode,
    "psn": ParallelSpikingNode,
    "tlif": TLIFNode,
    'ielif': IELIFNode,
    'ltmd': LTMD,
    'stbif': STBIF,
    'ilif': ILIFNeuron,
    'rplif': RPLIFNode,
    'izhikevich': IzhikevichNode,
}
