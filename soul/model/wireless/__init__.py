from .spikingtcn import SpikingTCN
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN
from .spikingvgg import SpikingVGG5, SpikingVGG9, SpikingVGG11, SpikingVGG13, SpikingVGG16, SpikingVGG19
from .sewresnet import SEWResNet18, SEWResNet34, SEWResNet50
from .msresnet import MSResNet18, MSResNet34, MSResNet50
from .spikformer import Spikformer256, Spikformer384
from .metaspikeformer import MetaSpikeformer256, MetaSpikeformer384
from .spikingresformer import SpikingResformer256, SpikingResformer384
from .qkformer import QKFormer256, QKFormer384

wireless_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'spikingtcn': SpikingTCN,
    'spikingvgg9': SpikingVGG9,
    'spikingvgg16': SpikingVGG16,
    'sewresnet34': SEWResNet34,
    'sewresnet50': SEWResNet50,
    'msresnet34': MSResNet34,
    'msresnet50': MSResNet50,
    'spikformer256': Spikformer256,
    'spikformer384': Spikformer384,
    'metaspikeformer256': MetaSpikeformer256,
    'metaspikeformer384': MetaSpikeformer384,
    'qkformer256': QKFormer256,
    'qkformer384': QKFormer384,
    'spikingresformer256': SpikingResformer256,
    'spikingresformer384': SpikingResformer384,
}