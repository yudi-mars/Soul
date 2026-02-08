欢迎来到Soul的文档
============================
EN document: :doc:`./index`

`Soul <https://github.com/yudi-mars/Soul>`_ 是一个基于Python和 `PyTorch <https://pytorch.org/>`_ 的开源工具包，
用于构建脉冲神经网络（SNN）应用程序。它提供了一个统一、高效的框架，既适用于研究，也适用于边缘部署，使您能够以最小的
开销重现受大脑启发的计算算法并开发新模型。借助SOUL（基于SNN的开源工具包），您可以在一个综合环境中无缝地试验SNN，该
环境将学术探索与现实世界的边缘智能紧密结合。

安装
----------------
注意，Soul是基于PyTorch的，需要确保环境中已经安装了PyTorch，才能安装Soul。
您可以从源代码安装最新版本：
通过 `GitHub <https://github.com/yudi-mars/Soul>`_ ：

.. code:: bash

    git clone https://github.com/yudi-mars/Soul.git

入门
----------------

.. toctree::
   :maxdepth: 2
   :caption: 目录
   
   getting_started

使用指南
----------------

.. toctree::
   :maxdepth: 2
   :caption: 目录

   concepts
   params
   neuron
   encoding
   datasets/index
   gradient
   models/index
   metrics
   app/index

编程指南
----------------

.. toctree::
   :maxdepth: 2
   :caption: 目录
   
   FAQs/add_dataset
   FAQs/add_coding
   FAQs/add_model
   FAQs/add_neuron
   FAQs/deployment

APIs
----------------

.. toctree::
   :maxdepth: 2
   :caption: 目录

   api/modules