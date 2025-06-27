from .spikingvgg import SpikingVGG5, SpikingVGG9, SpikingVGG11, SpikingVGG13, SpikingVGG16, SpikingVGG19
from .sewresnet import SEWResNet18, SEWResNet50, SEWResNet34
from .msresnet import MSResNet18, MSResNet34, MSResNet50
from .spikformer import Spikformer256, Spikformer384, Spikformer512
from .metaspikformer import MetaSpikformer256, MetaSpikformer384, MetaSpikformer512
from .spikingresformer import SpikingResformer192, SpikingResformer384, SpikingResformer512, SpikingResformer256
from .qkformer import QKFormer256, QKFormer384, QKFormer512

model_map = {
    'spikingvgg5': SpikingVGG5, 
    'spikingvgg9': SpikingVGG9, 
    'spikingvgg11': SpikingVGG11, 
    'spikingvgg13': SpikingVGG13, 
    'spikingvgg16': SpikingVGG16, 
    'spikingvgg19': SpikingVGG19, 
    'sewresnet18': SEWResNet18, 
    'sewresnet34': SEWResNet34, 
    'sewresnet50': SEWResNet50,
    'msresnet18': MSResNet18, 
    'msresnet34': MSResNet34, 
    'msresnet50': MSResNet50,
    'spikformer256': Spikformer256, 
    'spikformer384': Spikformer384, 
    'spikformer512': Spikformer512,
    'metaspikformer256': MetaSpikformer256, 
    'metaspikformer384': MetaSpikformer384, 
    'metaspikformer512': MetaSpikformer512,
    'spikingresformer192': SpikingResformer192, 
    'spikingresformer256': SpikingResformer256, 
    'spikingresformer384': SpikingResformer384, 
    'spikingresformer512': SpikingResformer512,
    'qkformer256': QKFormer256, 
    'qkformer384': QKFormer384, 
    'qkformer512': QKFormer512,
}
