from .dcnn import DCNN
from .sensehar import SenseHAR

motion_model_map = {
    'dcnn': DCNN,
    'sensehar': SenseHAR,
    
}