from .spikingvgg import SpikingVGG9, SpikingVGG16
from .sewresnet import SEWResNet18, SEWResNet50
from .msresnet import MSResNet18, MSResNet50
from .spikingtcn import SpikingTCN
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN

acoustic_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'spikingvgg9': SpikingVGG9, 
    'spikingvgg16': SpikingVGG16, 
    'sewresnet18': SEWResNet18, 
    'sewresnet50': SEWResNet50,
    'msresnet18': MSResNet18, 
    'msresnet50': MSResNet50,
    'spikingtcn': SpikingTCN,
}