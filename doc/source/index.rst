.. image:: _static/code-logo.jpg

Welcome to the Soul Documentation
============================
中文版文档: :doc:`./index_zh`

`Soul <https://github.com/yudi-mars/Soul>`_ is an open-source toolkit based on Python and `PyTorch <https://pytorch.org/>`_,
 designed for building Spiking Neural Network (SNN) applications. It provides a unified and efficient framework that 
 works for both research and edge deployment, allowing you to reproduce brain-inspired computing algorithms and develop 
 new models with minimal overhead. With Soul (the SNN-based Open-Source Utility Library), you can seamlessly experiment 
 with SNNs in an integrated environment that closely combines academic exploration with real-world edge intelligence.

Installation
----------------
Note that Soul is built on PyTorch. You must ensure PyTorch is already installed in your environment before installing Soul.
You can install the latest version from the source code via `GitHub <https://github.com/yudi-mars/Soul>`_:

.. code:: bash

    git clone https://github.com/yudi-mars/Soul.git


.. toctree::
   :maxdepth: 2
   :caption: GET_STARTED

   getting_started_en


.. toctree::
   :maxdepth: 2
   :caption: USER GUIDE

   concepts_en
   params_en
   neuron_en
   encoding_en
   datasets/index_en
   gradient_en
   models/index_en
   metrics_en
   app/index_en


.. toctree::
   :maxdepth: 2
   :caption: DEVELOPER GUIDE

   FAQs/add_dataset_en
   FAQs/add_coding_en
   FAQs/add_model_en
   FAQs/add_neuron_en
   FAQs/deployment


.. toctree::
   :maxdepth: 2
   :caption: APIs

   api/modules
