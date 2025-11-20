from .binary_parallel import encode as binary_par_encode
from .binary_sequential import encode as binary_seq_encode
from .bsa import encode as bsa_encode
from .burst import encode as burst_encode
from .direct import encode as direct_code
from .phase import encode as phase_encode
from .poisson import encode as poisson_encode
from .population import encode as population_encode
from .rank_order import encode as rank_order_encode
from .temporal import encode as temporal_encode
from .rate import encode as rate_encode
from .sdr import encode as sdr_encode
from .tcr_mw import encode as tcr_mwc_encode
from .tcr_sf import encode as tcr_sf_decode
from .tcr_tbr import encode as tcr_tbr_encode
from .temporal import encode as temporal_encode


coding_map = {
    'binary_parallel': binary_par_encode,
    'binary_sequential': binary_seq_encode,
    'bsa': bsa_encode,
    'burst': burst_encode,
    'direct': direct_code,
    'phase': phase_encode,
    'poisson': poisson_encode,
    'population': population_encode,
    'rank_order': rank_order_encode,
    'temporal': temporal_encode,
    'rate': rate_encode,
    'sdr': sdr_encode,
    'tcr_mw': tcr_mwc_encode,
    'tcr_sf': tcr_sf_decode,
    'tcr_tbr': tcr_tbr_encode,
}
