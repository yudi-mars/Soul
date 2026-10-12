"""
Tests for soul.neuron: every registered neuron is built from its shipped yaml config and checked for
forward/backward passes, state handling, step modes, devices, checkpoint keys and model integration.

    python -m pytest tests -q
"""
import copy
import io
import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from soul.model import model_map
from soul.neuron import BaseNode, DendriticBilinear, MemoryModule, functional, neuron_map
from soul.utils import set_v_threshold
from soul.utils.surrogate import surrogate_map

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'soul' / 'config'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 4
SHAPES = [(T, 2, 3, 5, 5), (T, 2, 6, 8), (T, 2, 7)]

NEURONS = list(neuron_map)
STATEFUL = [n for n in NEURONS if issubclass(neuron_map[n], MemoryModule)]
NODES = [n for n in NEURONS if issubclass(neuron_map[n], BaseNode)]

# parameter names are part of the checkpoint format
EXPECTED_STATE_KEYS = {
    'lif': [], 'clif': [], 'ilif': [], 'rplif': [], 'intlif': [], 'tlif': [], 'stbif': [], 'izhikevich': [],
    'ielif': [], 'plif': ['w'], 'ltmd': ['w'], 'psn': ['weight', 'bias'],
    'glif': ['alpha', 'beta', 'gamma', 'tau', 'v_threshold', 'linear_decay', 'v_subreset', 'conduct'],
    'dgn': ['equilibrium'], 'dlif': [], 'polara': [],
}
# overrides that make stochastic neurons deterministic, for tests that compare repeated runs
DETERMINISTIC = {'polara': {'noise': 0.}}


def neuron_config(name, **overrides):
    config = yaml.safe_load(open(CONFIG_DIR / 'neuron' / f'{name}.yaml'))
    config['time_step'] = T
    config.update(overrides)
    config['surrogate_function'] = surrogate_map[config['surrogate']]
    return config


def build(name, **overrides):
    return neuron_map[name](neuron_config(name, **overrides))


def build_deterministic(name, **overrides):
    return build(name, **{**DETERMINISTIC.get(name, {}), **overrides})


def randn(shape, seed=0, device='cpu'):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * 1.5 + 0.5).to(device)


def spike_levels(name, neuron, y):
    """Map the outputs of a neuron back to integer spike levels and return them with the allowed range."""
    if name in ('intlif', 'tlif'):
        return y * neuron.norm, (-1, 1) if name == 'tlif' else (0, 8)
    if name == 'ielif':
        return y * 4, (0, 4)
    if name == 'stbif':
        return y / neuron.q_threshold, (0, 1)
    return y, (0, 1)


def test_every_neuron_has_a_config():
    for name, cls in neuron_map.items():
        assert (CONFIG_DIR / 'neuron' / f'{name}.yaml').exists(), name
        assert issubclass(cls, torch.nn.Module)


@pytest.mark.parametrize('name', NEURONS)
def test_build_from_shipped_config(name):
    neuron = build(name)
    assert repr(neuron)
    assert list(neuron.state_dict()) == EXPECTED_STATE_KEYS[name]
    assert not list(neuron.buffers())


@pytest.mark.parametrize('shape', SHAPES, ids=str)
@pytest.mark.parametrize('name', NEURONS)
def test_forward_backward(name, shape):
    neuron = build(name).to(DEVICE)
    x = randn(shape, device=DEVICE).requires_grad_()
    y = neuron(x)

    assert y.shape == x.shape and y.dtype == x.dtype
    if name == 'polara':  # graded spikes
        assert ((y == 0) | ((0.5 <= y) & (y < 1))).all()
    else:
        levels, (low, high) = spike_levels(name, neuron, y.detach())
        assert torch.equal(levels, levels.round())
        assert low <= levels.min() and levels.max() <= high

    (y * randn(shape, seed=1, device=DEVICE)).sum().backward()
    assert x.grad.abs().sum() > 0
    for pname, p in neuron.named_parameters():
        assert p.grad is not None, pname


@pytest.mark.parametrize('name', STATEFUL)
def test_reset_restores_initial_state(name):
    neuron = build_deterministic(name)
    initial = dict(neuron.named_memories())
    x = randn(SHAPES[0])
    with torch.no_grad():
        y = neuron(x)
        assert not torch.equal(neuron(x), y), 'state should carry over until reset'
        functional.reset_net(neuron)
        assert dict(neuron.named_memories()) == initial
        assert torch.equal(neuron(x), y)


@pytest.mark.parametrize('name', STATEFUL)
def test_deepcopies_do_not_share_state(name):
    proto = build_deterministic(name)
    a, b = copy.deepcopy(proto), copy.deepcopy(proto)
    x = randn(SHAPES[0])
    with torch.no_grad():
        y = a(x)
        assert dict(b.named_memories()) == dict(proto.named_memories())
        assert torch.equal(b(x), y)


@pytest.mark.parametrize('name', NODES)
def test_single_step_matches_multi_step(name):
    proto = build_deterministic(name)
    multi, single = copy.deepcopy(proto), copy.deepcopy(proto)
    multi.step_mode, single.step_mode = 'm', 's'
    x = randn(SHAPES[1])
    with torch.no_grad():
        assert torch.equal(multi(x), torch.stack([single(x[t]) for t in range(T)]))


@pytest.mark.parametrize('name', NODES)
def test_store_v_seq(name):
    neuron = build(name, store_v_seq=True, step_mode='m')
    x = randn(SHAPES[0])
    with torch.no_grad():
        neuron(x)
    assert neuron.v_seq.shape == x.shape
    assert torch.equal(neuron.v_seq[-1], neuron.v)


def test_default_step_modes():
    for name in ('lif', 'plif', 'clif', 'ilif', 'rplif', 'stbif', 'glif', 'izhikevich', 'dgn', 'dlif', 'polara'):
        assert build(name).step_mode == 'm', name
    for name in ('intlif', 'tlif'):
        assert build(name).step_mode == 's', name
    with pytest.raises(ValueError):
        build('glif').step_mode = 's'


@pytest.mark.parametrize('name', ['intlif', 'tlif'])
def test_integer_neurons_quantise_every_step_independently(name):
    neuron = build(name)
    x = randn(SHAPES[0])
    with torch.no_grad():
        y = neuron(x)
        for t in range(T):
            functional.reset_net(neuron)
            assert torch.equal(neuron(x[t]), y[t])


def test_intlif_output():
    x = randn(SHAPES[0]) * 3
    with torch.no_grad():
        assert torch.equal(build('intlif')(x), torch.round(torch.clamp(x, 0, 8)) / 8)


@pytest.mark.parametrize('name', ['glif', 'psn'])
def test_time_step_mismatch_is_reported(name):
    with pytest.raises(ValueError, match='time steps'):
        build(name)(randn((T - 1, 2, 7)))


def test_ltmd_runs_every_time_step():
    neuron = build('ltmd')
    x = randn((T + 2, 2, 7)) * 2
    with torch.no_grad():
        y = neuron(x)
        threshold = neuron.w.tanh()

    u, o, expected = torch.zeros(2, 7), torch.zeros(2, 7), []
    for t in range(T + 2):
        u = 0.25 * u * (1 - o) + x[t]
        o = (u - threshold > 0).float()
        expected.append(o)
    assert torch.equal(y, torch.stack(expected))


def test_stbif_matches_reference_and_has_gradient():
    neuron = build('stbif', q_threshold=0.3)
    x = randn((T + 2, 2, 7)).requires_grad_()
    y = neuron(x)

    v, expected = torch.full((2, 7), 0.5), []
    with torch.no_grad():
        for t in range(T + 2):
            v = v + x[t] / 0.3
            spike = (v >= 1.).float()
            v = v - spike
            expected.append(spike * 0.3)
    assert torch.equal(y.detach(), torch.stack(expected))
    y.sum().backward()
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize('decay_input', [True, False])
@pytest.mark.parametrize('v_reset', [0., 0.2, None])
def test_lif_matches_reference(decay_input, v_reset):
    neuron = build('lif', tau=2., v_threshold=1., v_reset=v_reset, decay_input=decay_input)
    x = randn((8, 64))
    with torch.no_grad():
        y = neuron(x)

    rest = 0. if v_reset is None else v_reset
    v, expected = torch.full((64,), rest), []
    for t in range(8):
        v = v + (rest - v + x[t]) / 2. if decay_input else v + (rest - v) / 2. + x[t]
        spike = (v > 1.).float()
        v = v - spike if v_reset is None else (1. - spike) * v + spike * v_reset
        expected.append(spike)
    assert torch.equal(y, torch.stack(expected))


def test_rplif_raises_threshold_after_a_spike():
    x = torch.tensor([[3.], [3.], [3.], [0.]])
    with torch.no_grad():
        lif, rplif = build('lif'), build('rplif', alpha=1.5)
        assert lif(x).flatten().tolist() == [1., 1., 1., 0.]
        assert rplif(x).flatten().tolist() == [1., 0., 1., 0.]
        assert rplif.v_th.flatten().tolist() == [1.]


def test_rplif_follows_assigned_threshold():
    rplif = build('rplif', alpha=1.5)
    set_v_threshold(rplif, 0.5)
    x = torch.full((2, 1), 1.4)  # v = 0.7 after every charge, above 0.5 but below 0.5 * 1.5
    with torch.no_grad():
        assert rplif(x).flatten().tolist() == [1., 0.]
        functional.reset_net(rplif)
        assert rplif(x).flatten().tolist() == [1., 0.]


@pytest.mark.parametrize('name', ['lif', 'plif', 'clif', 'ilif', 'rplif', 'izhikevich'])
def test_soft_reset(name):
    with torch.no_grad():
        y = build(name, v_reset=None)(randn(SHAPES[0]))
    assert y.sum() > 0


@pytest.mark.parametrize('name', [n for n in NEURONS if n not in ('psn', 'ielif', 'ltmd')])
def test_set_v_threshold_survives_reset(name):
    neuron = build(name)
    set_v_threshold(neuron, 0.3)
    functional.reset_net(neuron)
    threshold = neuron.v_threshold.sigmoid() if name == 'glif' else neuron.v_threshold
    assert torch.as_tensor(threshold).item() == pytest.approx(0.3)


@pytest.mark.parametrize('conductance', [False, True])
def test_dgn_matches_reference(conductance):
    neuron = build('dgn')
    x, g = randn((8, 64)), randn((8, 64), seed=1)
    with torch.no_grad():
        y = neuron(x, g) if conductance else neuron(x)

    decay = math.exp(-1. / neuron.tau_s)
    d, d_c, v, expected = torch.zeros(64), torch.zeros(64), torch.zeros(64), []
    for t in range(8):
        if conductance:
            d, d_c = d * decay + g[t], d_c * decay + x[t]
        else:
            d = d * decay + x[t]
            d_c = 1. * d  # s-DGN: current = E * conductance trace with E = 1
        v = torch.sigmoid(1. - neuron.g_l - d) * v + d_c
        spike = (v > 1.).float()
        v = v - spike
        expected.append(spike)
    assert torch.equal(y, torch.stack(expected))


def test_dgn_two_inputs_in_both_step_modes():
    proto = build('dgn')
    multi, single = copy.deepcopy(proto), copy.deepcopy(proto)
    single.step_mode = 's'
    x, g = randn(SHAPES[1]), randn(SHAPES[1], seed=1)
    with torch.no_grad():
        assert torch.equal(multi(x, g), torch.stack([single(x[t], g[t]) for t in range(T)]))


def test_polara_matches_reference():
    neuron = build('polara', noise=0.)
    x = randn((8, 64))
    with torch.no_grad():
        y = neuron(x)

    v, theta, expected = torch.zeros(64), 1., []
    for t in range(8):
        lags = range(t, t + 5)
        u = v * (1. - 0.1 * sum(math.exp(-2. * i) for i in lags) + 0.1) + x[t]
        delta = u - theta
        overshoot = torch.clamp(delta * (1. + delta * torch.exp(-torch.clamp(delta, min=-10.))), max=2.)
        expected.append((delta > 0).float() * torch.sigmoid(u * overshoot))
        v = u * (1. + sum(d * math.exp(-(i - mu) ** 2 / (2 * s ** 2))
                          for i in lags for d, mu, s in ((-0.2, 1.5, 0.7), (-0.1, 4., 1.5), (-0.1, 6., 4.))))
        theta = theta * (1. + sum(0.001 * math.exp(-0.5 * i) for i in lags))
    assert torch.equal(y, torch.stack(expected))


def test_polara_noise_is_reproducible_with_a_seed():
    x = randn(SHAPES[0])
    outputs = []
    for seed in (0, 0, 1):
        torch.manual_seed(seed)
        with torch.no_grad():
            outputs.append(build('polara')(x))
    assert torch.equal(outputs[0], outputs[1]) and not torch.equal(outputs[0], outputs[2])


@pytest.mark.parametrize('shape', [(3, 5, 7), (3, 5, 4, 6)], ids=str)
def test_dendritic_bilinear(shape):
    layer = DendriticBilinear(shape[1], 4)
    x = torch.randn(shape).requires_grad_()
    y = layer(x)
    assert y.shape == (shape[0], 4, *shape[2:])

    position = (0, slice(None)) + (0,) * (len(shape) - 2)
    p, w = x[position], layer.weight * layer.mask
    torch.testing.assert_close(y[position], torch.einsum('i,oij,j->o', p, w, p) + layer.bias)
    y.sum().backward()
    assert (torch.diagonal(layer.weight.grad, dim1=1, dim2=2) == 0).all()  # no self-interaction terms
    assert list(layer.state_dict()) == ['weight', 'bias']


@pytest.mark.parametrize('model_name,name', [('lenet_dlif', 'dlif'), ('lenet_dgn', 'dgn')])
def test_lenet_variants_train_every_trainable_parameter(model_name, name):
    model = model_map['vision'][model_name](model_config('vision', model_name, name)).to(DEVICE)
    model(randn((T, 2, 3, 32, 32), device=DEVICE)).sum().backward()
    for pname, p in model.named_parameters():
        assert (p.grad is not None) == p.requires_grad, pname  # DDP rejects unused trainable parameters


def test_lenet_dgn_needs_dgn():
    with pytest.raises(ValueError, match='dgn'):
        model_map['vision']['lenet_dgn'](model_config('vision', 'lenet_dgn', 'lif'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not available')
@pytest.mark.parametrize('name', NEURONS)
def test_cuda_matches_cpu(name):
    proto = build_deterministic(name)
    cpu, gpu = copy.deepcopy(proto), copy.deepcopy(proto).cuda()
    x = randn(SHAPES[0])
    with torch.no_grad():
        y_cpu, y_gpu = cpu(x), gpu(x.cuda())
        for _, value in getattr(gpu, 'named_memories', dict)():
            assert not isinstance(value, torch.Tensor) or value.is_cuda
        functional.reset_net(gpu)
        assert torch.equal(gpu(x.cuda()), y_gpu)
    assert (~torch.isclose(y_cpu, y_gpu.cpu(), rtol=1e-4, atol=1e-5)).float().mean() < 1e-3


def test_onnx_export_marks_neurons():
    onnx = pytest.importorskip('onnx')
    model = torch.nn.Sequential(torch.nn.Linear(6, 6), build('lif'))
    buffer = io.BytesIO()
    torch.onnx.export(model, torch.randn(T, 1, 6), buffer, export_params=False, do_constant_folding=False,
                      input_names=['input'], output_names=['output'], dynamo=False)
    nodes = [n for n in onnx.load_from_string(buffer.getvalue()).graph.node if n.domain == 'soul']
    assert [n.op_type for n in nodes] == ['LIFNode']
    assert [a.name for a in nodes[0].attribute] == ['spike_shape']


APPS = {
    'vision': ({'input_channels': 3, 'input_height': 32, 'input_width': 32}, (3, 32, 32)),
    'acoustic': ({'input_dim': 128, 'input_channels': 128}, (128, 128)),
    'motion': ({'input_channels': 9, 'input_dim': 128}, (9, 128)),
    'wireless': ({'input_channels': 1, 'input_height': 32, 'input_width': 32}, (1, 32, 32)),
}
MODELS = [(app, model) for app in APPS for model in ('mlp', 'lenet')] + [
    ('vision', 'rnn'), ('vision', 'spikingvgg5'), ('vision', 'sewresnet18'), ('vision', 'spikformer256'),
    ('acoustic', 'metaspikeformer256'), ('motion', 'dcnn'), ('wireless', 'spikformer'),
    ('vision', 'lenet_dlif'), ('motion', 'lenet_dlif'),
]
# larger models are checked with one neuron per code path to keep the suite fast
HEAVY_MODELS = [('vision', 'qkformer256'), ('vision', 'spikingresformer192'), ('motion', 'dcl'),
                ('motion', 'sewresnet34'), ('wireless', 'spiliformer256')]
HEAVY_NEURONS = ['lif', 'glif', 'psn', 'tlif', 'stbif', 'ltmd']
MODEL_CASES = [(app, model, name) for app, model in MODELS for name in NEURONS] + \
              [(app, model, name) for app, model in HEAVY_MODELS for name in HEAVY_NEURONS] + \
              [(app, 'lenet_dgn', 'dgn') for app in APPS]


def model_config(app, model_name, neuron_name):
    config = yaml.safe_load(open(CONFIG_DIR / 'basic.yaml'))
    config.update(neuron_config(neuron_name, **DETERMINISTIC.get(neuron_name, {})))
    prefix = re.match(r'^([a-zA-Z]+)', model_name).group(1).lower()
    model_yaml = CONFIG_DIR / 'model' / app / f'{prefix}.yaml'
    if model_yaml.exists():
        config.update(yaml.safe_load(open(model_yaml)) or {})
    config.update(APPS[app][0])
    config.update({'application': app, 'model': model_name, 'neuron_type': neuron_name, 'num_classes': 5})
    config['neuron'] = neuron_map[neuron_name](config)
    return config


@pytest.mark.parametrize('app,model_name,name', MODEL_CASES, ids=[f'{a}-{m}-{n}' for a, m, n in MODEL_CASES])
def test_model_integration(app, model_name, name):
    model = model_map[app][model_name](model_config(app, model_name, name)).to(DEVICE)
    x = randn((T, 2, *APPS[app][1]), device=DEVICE)

    model.train()
    out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (2, 5)
    F.cross_entropy(out, torch.tensor([0, 1], device=DEVICE)).backward()

    model.eval()
    with torch.no_grad():
        out = model(x)
        torch.testing.assert_close(model(x), out)  # reset_net clears neuron states between calls
    assert torch.isfinite(out).all()


@pytest.mark.parametrize('app,model_name', [('vision', 'qkformer256'), ('motion', 'qkformer256'),
                                            ('vision', 'spiliformer256'), ('wireless', 'spiliformer256')])
def test_neurons_receive_time_major_inputs(app, model_name):
    model = model_map[app][model_name](model_config(app, model_name, 'lif')).to(DEVICE)
    leading = []
    for m in model.modules():
        if isinstance(m, BaseNode):
            m.register_forward_pre_hook(lambda module, inputs: leading.append(inputs[0].shape[0]))
    with torch.no_grad():
        model(randn((T, 3, *APPS[app][1]), device=DEVICE))  # batch size != T so that (T * B, ...) inputs are caught
    assert leading and set(leading) == {T}


@pytest.mark.parametrize('app,model_name', [('vision', 'sewresnet18'), ('motion', 'sewresnet50')])
def test_sewresnet_zero_init_residual(app, model_name):
    config = model_config(app, model_name, 'lif')
    config['zero_init_residual'] = True
    model = model_map[app][model_name](config).to(DEVICE)
    last_norms = [m.conv3[1] if type(m).__name__ == 'Bottleneck' else m.conv2[1]
                  for m in model.modules() if type(m).__name__ in ('BasicBlock', 'Bottleneck')]
    assert last_norms and all((norm.weight == 0).all() for norm in last_norms)
    assert model(randn((T, 2, *APPS[app][1]), device=DEVICE)).shape == (2, 5)
