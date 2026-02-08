如何添加神经元
============================

本教程将告诉您如何添加一个自定义的神经元算子到Soul项目中。

在Soul工具包中，一个神经元类最重要的方法就是 ``forward(x)`` 方法。在 ``soul.neuron`` 包中，我们提供了大量的现有神经元类，
您可以通过继承并修改这些神经元类来构建自己的神经元算子。当然，您也可以直接继承 ``soul.neuron.base.MemoryModule`` 类，它是 
大部分神经元类的基类，提供了一系列SNN神经元方法的基础实现，您可以在此基础上实现自己的神经元。

在实现自己的神经元之后，在 ``soul.neuron.__init__.py`` 中的 ``neuron_map`` 注册它，就可以在之后的训练过程中使用自己的神经元了。

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
    # add your neuron
    'neuron_name': My_Neuron,
}