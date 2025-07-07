from .rate import rate_coding
from .direct import direct_coding
from .temporal import temporal_coding

coding_map = {
    'direct': direct_coding,
    'rate': rate_coding,
    'temporal': temporal_coding,
    # TODO more coding methods for future work
}