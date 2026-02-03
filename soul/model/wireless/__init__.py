from .spikingtcn import SpikingTCN
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN
from .spikingvgg import SpikingVGG9, SpikingVGG16
from .sewresnet import SEWResNet18, SEWResNet34, SEWResNet50
from .msresnet import MSResNet18, MSResNet34, MSResNet50
from .spikformer import Spikformer256, Spikformer384, SpikformerPrototype
from .metaspikeformer import MetaSpikeformer256, MetaSpikeformer384, MetaSpikeformerPrototype
from .spikingresformer import SpikingResformer256, SpikingResformer384, SpikingResformerPrototype
from .qkformer import QKFormer256, QKFormer384, QKFormerPrototype

wireless_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'spikingtcn': SpikingTCN,
    'spikingvgg9': SpikingVGG9,
    'spikingvgg16': SpikingVGG16,
    'sewresnet18': SEWResNet18,
    'sewresnet34': SEWResNet34,
    'sewresnet50': SEWResNet50,
    'msresnet18': MSResNet18,
    'msresnet34': MSResNet34,
    'msresnet50': MSResNet50,
    'spikformer': SpikformerPrototype,
    'spikformer256': Spikformer256,
    'spikformer384': Spikformer384,
    'metaspikeformer' : MetaSpikeformerPrototype,
    'metaspikeformer256': MetaSpikeformer256,
    'metaspikeformer384': MetaSpikeformer384,
    'qkformer': QKFormerPrototype,
    'qkformer256': QKFormer256,
    'qkformer384': QKFormer384,
    'spikingresformer': SpikingResformerPrototype,
    'spikingresformer256': SpikingResformer256,
    'spikingresformer384': SpikingResformer384,
}