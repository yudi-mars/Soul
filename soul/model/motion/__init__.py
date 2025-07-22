from .dcnn import DCNN
from .sensehar import SenseHAR
from .dcl import DCL

motion_model_map = {
    'dcnn': DCNN,
    'sensehar': SenseHAR,
    'dcl': DCL,
}