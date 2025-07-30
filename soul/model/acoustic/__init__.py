# from .spikingvgg import SpikingVGG5, SpikingVGG9, SpikingVGG11, SpikingVGG13, SpikingVGG16, SpikingVGG19
# from .sewresnet import SEWResNet18, SEWResNet50, SEWResNet34
# from .msresnet import MSResNet18, MSResNet34, MSResNet50
from .spikingtcn import SpikingTCN

acoustic_model_map = {
    # 'spikingvgg5': SpikingVGG5, 
    # 'spikingvgg9': SpikingVGG9, 
    # 'spikingvgg11': SpikingVGG11, 
    # 'spikingvgg13': SpikingVGG13, 
    # 'spikingvgg16': SpikingVGG16, 
    # 'spikingvgg19': SpikingVGG19, 
    # 'sewresnet18': SEWResNet18, 
    # 'sewresnet34': SEWResNet34, 
    # 'sewresnet50': SEWResNet50,
    # 'msresnet18': MSResNet18, 
    # 'msresnet34': MSResNet34, 
    # 'msresnet50': MSResNet50,
    'spikingtcn': SpikingTCN,
}