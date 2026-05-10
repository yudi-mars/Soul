from .spikingvgg import SpikingVGG9, SpikingVGG16
from .sewresnet import SEWResNet34, SEWResNet50
from .msresnet import MSResNet34, MSResNet50
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN
from .spikformer import Spikformer256, Spikformer384, Spikformer512
from .spikingformer import Spikingformer256, Spikingformer384, Spikingformer512
from .metaspikeformer import MetaSpikeformer256, MetaSpikeformer384, MetaSpikeformer512
from .spikingresformer import SpikingResformer192, SpikingResformer384, SpikingResformer512, SpikingResformer256
from .qkformer import QKFormer256, QKFormer384, QKFormer512
from .spiliformer import SpiLiFormer256, SpiLiFormer384, SpiLiFormer512

acoustic_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'spikingvgg9': SpikingVGG9, 
    'spikingvgg16': SpikingVGG16, 
    'sewresnet34': SEWResNet34, 
    'sewresnet50': SEWResNet50,
    'msresnet34': MSResNet34, 
    'msresnet50': MSResNet50,
    'spikformer256': Spikformer256,
    'spikformer384': Spikformer384,
    'spikformer512': Spikformer512,
    'spikingformer256': Spikingformer256,
    'spikingformer384': Spikingformer384,
    'spikingformer512': Spikingformer512,
    'metaspikeformer256': MetaSpikeformer256, 
    'metaspikeformer384': MetaSpikeformer384, 
    'metaspikeformer512': MetaSpikeformer512,
    'spikingresformer192': SpikingResformer192, 
    'spikingresformer256': SpikingResformer256, 
    'spikingresformer384': SpikingResformer384, 
    'spikingresformer512': SpikingResformer512,
    'qkformer256': QKFormer256, 
    'qkformer384': QKFormer384, 
    'qkformer512': QKFormer512,
    'spiliformer256': SpiLiFormer256, 
    'spiliformer384': SpiLiFormer384, 
    'spiliformer512': SpiLiFormer512,
}