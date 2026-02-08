How to Add a Neuron
============================

This tutorial will show you how to add a custom neuron operator to the Soul project.

In the Soul toolkit, the most important method of a neuron class is the ``forward(x)`` method. Within the ``soul.neuron`` package, we provide a large number of existing neuron classes.
You can build your own neuron operator by inheriting from and modifying these existing neuron classes. Alternatively, you can directly inherit the ``soul.neuron.base.MemoryModule`` class, which serves as the base class for most neuron classes and provides basic implementations for a series of SNN (Spiking Neural Network) neuron methods, allowing you to implement your custom neuron on top of it.

After implementing your custom neuron, register it in the ``soul.neuron.__init__.py`` within ``neuron_map`` , and you will be able to use your neuron in subsequent training processes.

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