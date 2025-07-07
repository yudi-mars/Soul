from .rectangular import Rectangular, Rectangle
from .sigmoid import Sigmoid, FastSigmoid
from .atan import ATan
from .quadratic import PieceWiseQuadratic
from .exp import PieceWiseExp
from .softsign import SoftSign
from .superspike import SuperSpike
from .erf import Erf
from .qpseudo import QPseudoSpike
from .ternary import Ternary
from .quantize import Quant, Quant4

surrogate_map = {
    'atan': ATan(),
    'erf': Erf(),
    'sigmoid': Sigmoid(),
    'rectangle': Rectangle(),
    'fastsigmoid': FastSigmoid(),
    'qpseudospike': QPseudoSpike(),
    'softsign': SoftSign(),
    'quandratic': PieceWiseQuadratic(),
    'exp': PieceWiseExp(),
    'superspike': SuperSpike(),
    'ternary': Ternary(),
    'quant': Quant(),
    'quant4': Quant4(), 
}