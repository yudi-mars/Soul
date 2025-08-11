from .spikingtcn import SpikingTCN
from .general import SpikingMLP, SpikingLeNet, SpikingRNN, SpikingConvRNN

wireless_model_map = {
    'mlp': SpikingMLP,
    'lenet': SpikingLeNet,
    'rnn': SpikingRNN,
    'convrnn': SpikingConvRNN,
    'spikingtcn': SpikingTCN,
}