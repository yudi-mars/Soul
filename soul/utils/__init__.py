from .data import load_data, get_loader
from .metrics import accuracy, AverageMeter, ops_monitor, MODULE_SOP_DICT
from .parser import init_config
from .random import init_seed
from .log import setup_logger
from .utility import *
from .surrogate import *
from .power_check import *
from .coding import direct_coding, rate_coding, temporal_coding

surrogate_map = {
    'atan': ATan(),
    'erf': Erf(),
    'rect': Rectangular(),
    'sigmoid': FastSigmoid(),
    'ternary': Ternary(),
    'quant': Quant(),
    'quant4': Quant4(),
    'rectangle': Rectangle(),
}

coding_map = {
    'direct': direct_coding,
    'rate': rate_coding,
    'temporal': temporal_coding,
    # TODO more coding methods
}
