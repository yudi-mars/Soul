如何添加神经元
============================

本教程将告诉您如何添加一个自定义的神经元算子到Soul项目中。

``soul.neuron`` 中的神经元按以下层次组织：

.. code-block:: text

    MemoryModule                      带状态的模块，reset() 时恢复状态（膜电位等）
    ├── BaseNode                      充电 -> 发放 -> 重置 的逐时间步模板
    │   ├── LIFNode
    │   │   ├── CLIFNode
    │   │   ├── ILIFNeuron
    │   │   ├── RPLIFNode
    │   │   └── DLIFNode
    │   ├── ParametricLIFNode
    │   ├── INTLIFNode
    │   │   └── TLIFNode
    │   ├── STBIF
    │   ├── DGNNode
    │   ├── POLARANode
    │   └── AdaptBaseNode
    │       └── IzhikevichNode
    └── GatedLIFNode                  只支持多步模式
    nn.Module（无状态，一次处理整段序列）
    ├── ParallelSpikingNode
    ├── IELIFNode
    └── LTMD

所有神经元都通过 ``neuron_map[name](config)`` 由全局配置字典构造。``soul.neuron.base.BaseNode`` 负责读取公共配置项
（``v_threshold`` 、 ``v_reset`` 、 ``surrogate_function`` 、 ``detach_reset`` 、 ``store_v_seq`` 、 ``step_mode`` ），
并实现每个时间步的 ``neuronal_charge(x)`` -> ``neuronal_fire()`` -> ``neuronal_reset(spike)`` 流程以及单步/多步推理。
大多数新神经元只需要继承 ``BaseNode`` （或与之最接近的现有神经元，如 ``LIFNode`` ），在 ``__init__`` 中读取额外的配置项，
并实现 ``neuronal_charge`` ；需要额外状态时用 ``register_memory`` 注册，它们会在 ``functional.reset_net`` 时自动恢复初值。

.. code-block:: python

    import torch
    from soul.neuron.base import BaseNode

    class MyNode(BaseNode):
        def __init__(self, config):
            super().__init__(config)
            self.tau = config['tau']

        def neuronal_charge(self, x: torch.Tensor):
            self.v = self.v + (x - self.v) / self.tau

如果神经元的动力学不符合逐时间步的模板（例如需要在整段序列上计算），可以直接继承 ``soul.neuron.base.MemoryModule`` 并实现
``multi_step_forward`` （参考 ``GatedLIFNode`` ），不保存状态的神经元则可以直接继承 ``torch.nn.Module`` 。

实现神经元之后，在 ``soul/config/neuron/`` 下添加同名的yaml配置文件（必须包含 ``surrogate`` 字段），并在 ``soul.neuron.__init__.py``
中的 ``neuron_map`` 注册它，就可以在之后的训练过程中通过 ``-n neuron_name`` 使用自己的神经元了。

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
        'dgn': DGNNode,
        'dlif': DLIFNode,
        'polara': POLARANode,
        # add your neuron
        'neuron_name': MyNode,
    }

最后运行 ``python -m pytest tests`` ，测试会自动覆盖 ``neuron_map`` 中注册的所有神经元。
