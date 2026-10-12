"""
Spiking neurons of Soul. Every neuron is built from the global config dict: ``neuron_map[name](config)``.

    MemoryModule                      stateful module, memories restored by reset()
    ├── BaseNode                      charge -> fire -> reset template
    │   ├── LIFNode
    │   │   ├── CLIFNode
    │   │   ├── ILIFNeuron
    │   │   ├── RPLIFNode
    │   │   └── DLIFNode              dendritic bilinear integration: DendriticBilinear in lenet_dlif
    │   ├── ParametricLIFNode
    │   ├── INTLIFNode
    │   │   └── TLIFNode
    │   ├── STBIF
    │   ├── DGNNode                   optional conductance input: full DGN in lenet_dgn
    │   ├── POLARANode
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
from .DLIF import DLIFNode, DendriticBilinear
from .PLIF import ParametricLIFNode
from .INTLIF import INTLIFNode
from .TLIF import TLIFNode
from .STBIF import STBIF
from .DGN import DGNNode
from .POLARA import POLARANode
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
    'dgn': DGNNode,
    'dlif': DLIFNode,
    'polara': POLARANode,
}
