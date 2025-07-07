from .CLIF import CLIFNode
from .GLIF import GatedLIFNode
from .ILIF import ILIFNode
from .LIF import LIFNode
from .PLIF import ParametricLIFNode
from .PSN import ParallelSpikingNode
from .TLIF import TLIFNode
from .IELIF import IELIFNode
from .LTMD import LTMD
# from .SRIF import SRIFLIFNode

neuron_map = {
    "lif": LIFNode,
    "plif": ParametricLIFNode,
    "clif": CLIFNode,
    "glif": GatedLIFNode,
    "ilif": ILIFNode,
    "psn": ParallelSpikingNode,
    "tlif": TLIFNode,
    'ielif': IELIFNode,
    'ltmd': LTMD,
    # 'srif': SRIFLIFNode
    # TODO
}