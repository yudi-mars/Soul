from .bsa import encode as bsa_encode
from .burst import encode as burst_encode
from .direct import encode as direct_code
from .phase import encode as phase_encode
from .poisson import encode as poisson_encode
from .population import encode as population_encode
from .rank_order import encode as rank_order_encode
from .rate import encode as rate_encode
from .ttfs import encode as ttfs_encode
from .tsc import encode as tsc_encode

coding_map = {
    'bsa': bsa_encode,
    'burst': burst_encode,
    'direct': direct_code,
    'phase': phase_encode,
    'poisson': poisson_encode,
    'population': population_encode,
    'rank_order': rank_order_encode,
    'ttfs': ttfs_encode,
    'rate': rate_encode,
    'tsc': tsc_encode,
}
'''
binary_parallel、binary_sequential、bsa、burst、phase、rank_order、ttfs、sdr、tcr_mw、tcr_sf、tcr_tbr、tsc
direct、poisson、population、rate、
'''