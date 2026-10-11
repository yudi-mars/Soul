How to Add a Neuron
============================

This tutorial will show you how to add a custom neuron operator to the Soul project.

The neurons in ``soul.neuron`` are organized as follows:

.. code-block:: text

    MemoryModule                      stateful module, states (e.g. membrane potential) are restored by reset()
    ├── BaseNode                      per-time-step charge -> fire -> reset template
    │   ├── LIFNode
    │   │   ├── CLIFNode
    │   │   ├── ILIFNeuron
    │   │   └── RPLIFNode
    │   ├── ParametricLIFNode
    │   ├── INTLIFNode
    │   │   └── TLIFNode
    │   ├── STBIF
    │   └── AdaptBaseNode
    │       └── IzhikevichNode
    └── GatedLIFNode                  multi-step only
    nn.Module (stateless, whole sequence at once)
    ├── ParallelSpikingNode
    ├── IELIFNode
    └── LTMD

Every neuron is built from the global config dict through ``neuron_map[name](config)``. ``soul.neuron.base.BaseNode`` reads the
shared configurations (``v_threshold``, ``v_reset``, ``surrogate_function``, ``detach_reset``, ``store_v_seq``, ``step_mode``) and
implements the per-time-step ``neuronal_charge(x)`` -> ``neuronal_fire()`` -> ``neuronal_reset(spike)`` flow as well as single-step and
multi-step inference. Most new neurons only need to inherit ``BaseNode`` (or the closest existing neuron such as ``LIFNode``), read their
extra configurations in ``__init__`` and implement ``neuronal_charge``. Extra states should be registered with ``register_memory`` so that
``functional.reset_net`` restores their initial values.

.. code-block:: python

    import torch
    from soul.neuron.base import BaseNode

    class MyNode(BaseNode):
        def __init__(self, config):
            super().__init__(config)
            self.tau = config['tau']

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + (x - self.v) / self.tau

If the dynamics do not fit the per-time-step template (e.g. they are computed over the whole sequence), inherit
``soul.neuron.base.MemoryModule`` directly and implement ``multi_step_forward`` (see ``GatedLIFNode``); stateless neurons can simply
inherit ``torch.nn.Module``.

After implementing your custom neuron, add a yaml config file with the same name under ``soul/config/neuron/`` (it must contain the
``surrogate`` field) and register the neuron in ``neuron_map`` of ``soul.neuron.__init__.py``. You will then be able to use it in
subsequent training processes with ``-n neuron_name``.

.. code-block:: python

    neuron_map = {
        "lif": LIFNode,
        "plif": ParametricLIFNode,
        "clif": CLIFNode,
        "glif": GatedLIFNode,
        "intlif": INTLIFNode,
        "psn": ParallelSpikingNode,
        "tlif": TLIFNode,
        'ielif': IELIFNode,
        'ltmd': LTMD,
        'stbif': STBIF,
        'ilif': ILIFNeuron,
        'rplif': RPLIFNode,
        'izhikevich': IzhikevichNode,
        # add your neuron
        'neuron_name': MyNode,
    }

Finally, run ``python -m pytest tests``; the tests automatically cover every neuron registered in ``neuron_map``.
