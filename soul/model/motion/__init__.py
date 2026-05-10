from .dcnn import DCNN
from .metaspikeformer import MetaSpikeformer256, MetaSpikeformer384
from .msresnet import MSResNet34, MSResNet50
from .qkformer import QKFormer256, QKFormer384
from .sensehar import SenseHAR
from .dcl import DCL
from .spikformer import Spikformer256, Spikformer384
from .spikingformer import Spikingformer256, Spikingformer384, Spikingformer512
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN
from .sewresnet import SEWResNet34, SEWResNet50
from .spikingresformer import SpikingResformer256, SpikingResformer384
from .spikingvgg import SpikingVGG9, SpikingVGG16
from .spiliformer import SpiLiFormer256, SpiLiFormer384, SpiLiFormer512

motion_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'dcnn': DCNN,
    'sensehar': SenseHAR,
    'dcl': DCL,
    'spikformer256': Spikformer256,
    'spikformer384': Spikformer384,
    'spikingformer256': Spikingformer256,
    'spikingformer384': Spikingformer384,
    'spikingformer512': Spikingformer512,
    'spikingvgg9': SpikingVGG9,
    'spikingvgg16': SpikingVGG16,
    'sewresnet34': SEWResNet34,
    'sewresnet50': SEWResNet50,
    'msresnet34': MSResNet34,
    'msresnet50': MSResNet50,
    'metaspikeformer256': MetaSpikeformer256,
    'metaspikeformer384': MetaSpikeformer384,
    'spikingresformer256': SpikingResformer256,
    'spikingresformer384': SpikingResformer384,
    'qkformer256': QKFormer256,
    'qkformer384': QKFormer384,
    'spiliformer256': SpiLiFormer256, 
    'spiliformer384': SpiLiFormer384, 
    'spiliformer512': SpiLiFormer512,
}