from .dcnn import DCNN
from .sensehar import SenseHAR
from .dcl import DCL
from .ispikformer import ISpikformer
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN

motion_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'dcnn': DCNN,
    'sensehar': SenseHAR,
    'dcl': DCL,
    'ispikformer': ISpikformer,
}