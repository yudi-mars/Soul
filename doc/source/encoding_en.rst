Encoding Methods
================

This chapter introduces the encoding methods supported by the Soul toolkit. The encoder maps external inputs (images, audio, sensors, time series, etc.) into a “time-first” sequence format so that the subsequent SNN can process them on discrete time steps.

**Unified conventions**
- Let the input tensor be :math:`\mathbf{x}`, and the encoding length be :math:`T=\texttt{num_steps}`.
- Inside the encoder, the input is first converted to ``float32`` and min–max normalized to :math:`[0,1]`: denoted as :math:`\tilde{\mathbf{x}}\in[0,1]`.
- The output uses a time-first dimension order: the output shape is ``(T, *inputs.shape)``.

bsa
----------------
Overview:
    Ben’s Spiker Algorithm with shape-aware routing (Ben's Spiker Algorithm, BSA). BSA is closer to a class of “sparsify a continuous signal into a spike train while being able to reconstruct it with a fixed kernel” matching-pursuit process: it fits the input with a causal kernel (causal kernel); when the residual exceeds a threshold, it emits a spike and updates the residual. It is suitable for continuous waveforms such as audio/sensors (and can also process images/feature maps independently per channel).

2D input mode formula:

.. math::

    s(t)=\mathbb{1}\big[e(t)>\theta\big],\quad e(t)=x(t)-\sum_{\tau<t} s(\tau)\,h(t-\tau)

where :math:`h` is an exponentially decaying kernel, :math:`\theta` is an adaptive threshold, and :math:`e(t)` is the current residual.

Additional implementation details (2D mode): let

.. math::

    \mathrm{corr}(w,c)=\sum_{\tau=0}^{L-1}\,\tilde x_{w+\tau,c}\,h(\tau),\quad
    a=\mathrm{clamp}\Big(\frac{\mathrm{corr}-\theta}{c_{\max}-\theta},0,1\Big),
    \;t^*=\left\lfloor (T-1)(1-a)\right\rfloor.

3D input mode formula:

.. math::

    t^* = \left\lfloor\frac{\mathrm{rank}(\tilde x)}{P-1}(T-1)\right\rfloor,\quad s_t=\mathbb{1}[t=t^*]

where :math:`\mathrm{rank}(\tilde x)\in\{0,\dots,P-1\}` is the index in descending order (the maximum value has rank=0), and :math:`P=H\times W` is the total number of spatial positions.

Implementation notes:

- For 2D inputs, an adaptive-threshold adjustment is used; the target spike-density range is [0.10, 0.35] 
- For 3D inputs, the actual implementation uses a rank-order coding strategy
References:

    Schrauwen & Van Campenhout, IJCNN 2003 (BSA)
    Auge et al., Neural Processing Letters 2021 (Survey)

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//bsa/T_00.png" alt="bsa T_00" style="width:150px;"/>
         <img src="_static/visible//bsa/T_01.png" alt="bsa T_01" style="width:150px;"/>
         <img src="_static/visible//bsa/T_02.png" alt="bsa T_02" style="width:150px;"/>
         <img src="_static/visible//bsa/T_03.png" alt="bsa T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//bsa/T_04.png" alt="bsa T_04" style="width:150px;"/>
         <img src="_static/visible//bsa/T_05.png" alt="bsa T_05" style="width:150px;"/>
         <img src="_static/visible//bsa/T_06.png" alt="bsa T_06" style="width:150px;"/>
         <img src="_static/visible//bsa/T_07.png" alt="bsa T_07" style="width:150px;"/>
     </div>

burst
----------------
Overview:
    Burst coding. It maps input intensity to a short burst packet: the stronger the input, the denser the spikes within the same window (smaller ISI). Each element emits at most B_max spikes, and the inter-spike interval is determined by the input intensity. It supports 2D and 3D inputs, and the output has the same dimensions as the input.

Formula:

.. math::

    B_{\max} = \min(4,T),\quad
    \mathrm{ISI}_{\max} = \max\left(1, \frac{T-1}{B_{\max}-1}\right),

.. math::

    B(\tilde x)=1+\left\lfloor\tilde x\,(B_{\max}-1)\right\rfloor,
    \;\mathrm{ISI}(\tilde x)=\mathrm{round}\big(\mathrm{ISI}_{\max}-(\mathrm{ISI}_{\max}-1)\,\tilde x\big),

.. math::

    t_k=k\cdot\mathrm{ISI}(\tilde x),\quad s(t)=\sum_{k=0}^{B(\tilde x)-1} \mathbb{1}[t=t_k].

Where:

- :math:`\tilde x` is the input value normalized to [0,1]
- :math:`B_{\max}` is the maximum number of spikes (no more than 4)
- :math:`\mathrm{ISI}_{\max}` is the maximum inter-spike interval
- :math:`B(\tilde x)` is the actual number of spikes
- :math:`\mathrm{ISI}(\tilde x)` is the actual inter-spike interval
Implementation notes:

- ISI is constrained to be at least 1 to ensure spikes do not overlap
- Spike times are computed cumulatively: t_k = k * ISI
References:

    Guo et al., "Neural Coding in Spiking Neural Networks: A Comprehensive Review", Frontiers in Neuroscience, 2021 (burst definition & properties)

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//burst/T_00.png" alt="burst T_00" style="width:150px;"/>
         <img src="_static/visible//burst/T_01.png" alt="burst T_01" style="width:150px;"/>
         <img src="_static/visible//burst/T_02.png" alt="burst T_02" style="width:150px;"/>
         <img src="_static/visible//burst/T_03.png" alt="burst T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//burst/T_04.png" alt="burst T_04" style="width:150px;"/>
         <img src="_static/visible//burst/T_05.png" alt="burst T_05" style="width:150px;"/>
         <img src="_static/visible//burst/T_06.png" alt="burst T_06" style="width:150px;"/>
         <img src="_static/visible//burst/T_07.png" alt="burst T_07" style="width:150px;"/>
     </div>

direct
----------------
Overview:
    Direct coding (Direct). It copies the normalized input T times and uses it directly as the driving signal at each time step.

Formula:

.. math::

   u(t)=\tilde x,\quad \forall t\in\{0,\dots,T-1\}

References:

    Youngeun Kim, "Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?" IEEE 2022.

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//direct/T_00.png" alt="direct T_00" style="width:150px;"/>
         <img src="_static/visible//direct/T_01.png" alt="direct T_01" style="width:150px;"/>
         <img src="_static/visible//direct/T_02.png" alt="direct T_02" style="width:150px;"/>
         <img src="_static/visible//direct/T_03.png" alt="direct T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//direct/T_04.png" alt="direct T_04" style="width:150px;"/>
         <img src="_static/visible//direct/T_05.png" alt="direct T_05" style="width:150px;"/>
         <img src="_static/visible//direct/T_06.png" alt="direct T_06" style="width:150px;"/>
         <img src="_static/visible//direct/T_07.png" alt="direct T_07" style="width:150px;"/>
     </div>

phase
----------------
Overview:
    Binary phase coding (Phase coding). It expands an input value normalized to [0,1] into an 8-bit binary fraction and maps each bit to a spike on a phase (time step). The first 8 time steps use decreasing weights (1, 1/2, 1/4, ...); for more than 8 steps, it cyclically repeats the phase pattern of the first 8 bits.
Formula:

.. math::
    q = \left\lfloor\tilde x\cdot 256\right\rfloor,\quad
    s_t = \left((q \gg (8-t))\;\bmod\;2\right),\quad
    o_t = s_t\;2^{-(t-1)}\quad (t=1,\dots,8)

.. math::
    o_t = o_{t \bmod 8}\quad (t>8)

Where:

- :math:`\tilde x` is the input value normalized to [0,1]
- :math:`q` is the integer quantized to 8 bits
- :math:`s_t` is the binary value of the t-th bit
- :math:`o_t` is the output value at the t-th time step (with weight)
Implementation notes:

- The weights of the first 8 time steps decrease as negative powers of 2
- For more than 8 steps, the pattern of the first 8 bits is repeated cyclically
References:

    Hwang & Kung, "One-Spike SNN: Single-Spike Phase Coding with Complex Synapse Accumulation for ANN-to-SNN Conversion Loss Minimization", IEEE 2025.

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//phase/T_00.png" alt="phase T_00" style="width:150px;"/>
         <img src="_static/visible//phase/T_01.png" alt="phase T_01" style="width:150px;"/>
         <img src="_static/visible//phase/T_02.png" alt="phase T_02" style="width:150px;"/>
         <img src="_static/visible//phase/T_03.png" alt="phase T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//phase/T_04.png" alt="phase T_04" style="width:150px;"/>
         <img src="_static/visible//phase/T_05.png" alt="phase T_05" style="width:150px;"/>
         <img src="_static/visible//phase/T_06.png" alt="phase T_06" style="width:150px;"/>
         <img src="_static/visible//phase/T_07.png" alt="phase T_07" style="width:150px;"/>
     </div>

population
----------------
Overview:
    Population coding. It uses Gaussian tuning curves to map an input value to a set of population responses. Each neuron has a maximal response over a specific input range, and the response strength is determined by a Gaussian function. In the implementation, tuning centers are uniformly distributed in [0,1], and beta=100.0 controls the sharpness of the response. It supports 2D and 3D inputs; the output dimension is determined by num_steps.

Formula:

.. math::

    y_i = \exp\big(-\beta(\tilde x-\mu_i)^2\big),\quad i=1,\dots,N

Where:

- :math:`\mu_i` is the tuning center of the i-th neuron, uniformly distributed in [0,1]
- :math:`\beta=100.0` controls the sharpness of the response
References:

    Norse docs: norse.torch.module.encode.PopulationEncoder
      https://norse.github.io/norse/generated/norse.torch.module.encode.PopulationEncoder.html
    Norse docs: norse.torch.functional.encode (population encoding)
      https://norse.github.io/norse/auto_api/norse.torch.functional.encode.html

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//population/T_00.png" alt="population T_00" style="width:150px;"/>
         <img src="_static/visible//population/T_01.png" alt="population T_01" style="width:150px;"/>
         <img src="_static/visible//population/T_02.png" alt="population T_02" style="width:150px;"/>
         <img src="_static/visible//population/T_03.png" alt="population T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//population/T_04.png" alt="population T_04" style="width:150px;"/>
         <img src="_static/visible//population/T_05.png" alt="population T_05" style="width:150px;"/>
         <img src="_static/visible//population/T_06.png" alt="population T_06" style="width:150px;"/>
         <img src="_static/visible//population/T_07.png" alt="population T_07" style="width:150px;"/>
     </div>

rank_order
----------------
Overview:
    Rank-Order coding. For each spatial position (or sample position), it sorts along the channel dimension; the rank level is mapped to a discrete time step, and it emits one spike at that time step. The larger the intensity, the earlier the rank, and thus the earlier the firing.

.. math::

    \hat x = \tilde x + \epsilon\frac{i}{N-1},\quad
    t^* = \left\lfloor\frac{\mathrm{rank}(\hat x)}{N-1}(T-1)\right\rfloor,\quad s_t=\mathbb{1}[t=t^*]

Where:

- :math:`\epsilon` is a small constant (1e-6 in the code)
- :math:`i` is the element index
- :math:`N` is the total number of elements
- :math:`\mathrm{rank}(\hat x)\in\{0,\dots,N-1\}` is the position index in descending order (the maximum value has rank=0)
Implementation notes:

- A deterministic shift is used to handle equal values, avoiding spike crowding caused by many ties
- Each element emits exactly one spike (one-spike-per-element)

References:

    Alan Jeffares et al., "Spike-inspired rank coding for fast and accurate recurrent neural networks", Proc. ICLR 2022. https://github.com/codingrank

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//rank_order/T_00.png" alt="rank_order T_00" style="width:150px;"/>
         <img src="_static/visible//rank_order/T_01.png" alt="rank_order T_01" style="width:150px;"/>
         <img src="_static/visible//rank_order/T_02.png" alt="rank_order T_02" style="width:150px;"/>
         <img src="_static/visible//rank_order/T_03.png" alt="rank_order T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//rank_order/T_04.png" alt="rank_order T_04" style="width:150px;"/>
         <img src="_static/visible//rank_order/T_05.png" alt="rank_order T_05" style="width:150px;"/>
         <img src="_static/visible//rank_order/T_06.png" alt="rank_order T_06" style="width:150px;"/>
         <img src="_static/visible//rank_order/T_07.png" alt="rank_order T_07" style="width:150px;"/>
     </div>

rate
----------------
Overview:
    Rate coding, also known as Poisson coding, maps the input intensity to the number of discharges/probability within a unit time window (i.e., the pulse rate), and represents the input intensity by the number of discharges/probability within a unit time window.
Formula:

.. math::

   s_t \sim \mathrm{Bernoulli}(p=\tilde x),\quad t=0,\dots,T-1


References:

    JK Eshraghian et al., "Training Spiking Neural Networks Using Lessons from Deep Learning", Proc. IEEE 2023. https://github.com/jeshraghian/snntorch

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//rate/T_00.png" alt="rate T_00" style="width:150px;"/>
         <img src="_static/visible//rate/T_01.png" alt="rate T_01" style="width:150px;"/>
         <img src="_static/visible//rate/T_02.png" alt="rate T_02" style="width:150px;"/>
         <img src="_static/visible//rate/T_03.png" alt="rate T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//rate/T_04.png" alt="rate T_04" style="width:150px;"/>
         <img src="_static/visible//rate/T_05.png" alt="rate T_05" style="width:150px;"/>
         <img src="_static/visible//rate/T_06.png" alt="rate T_06" style="width:150px;"/>
         <img src="_static/visible//rate/T_07.png" alt="rate T_07" style="width:150px;"/>
     </div>

tsc
----------------
Overview:
    Temporal Switch Coding (TSC). It emphasizes preserving the temporal structure of the input: if the input itself contains a time axis (such as raw sensor sequences, skeletal sequences, event streams), TSC often first converts this time axis to time-first and then performs a lightweight amplitude-to-pulse mapping (such as thresholding or rate sampling) at each time step.
Formula:

.. math::

    p^*=\left\lfloor \tilde p\,(T-1)\right\rfloor+1,
    \quad \text{if }p^*\ge 2:\; s_{0}=\operatorname{sign}(\tilde p),\; s_{p^*-1}=-\operatorname{sign}(\tilde p),
    \quad \text{otherwise no spikes}.

References:

    Han et al., "Deep Spiking Neural Network: Energy Efficiency Through Time-Based Coding", ECCV, 2020. (TSC definition & properties)

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//tsc/T_00.png" alt="tsc T_00" style="width:150px;"/>
         <img src="_static/visible//tsc/T_01.png" alt="tsc T_01" style="width:150px;"/>
         <img src="_static/visible//tsc/T_02.png" alt="tsc T_02" style="width:150px;"/>
         <img src="_static/visible//tsc/T_03.png" alt="tsc T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//tsc/T_04.png" alt="tsc T_04" style="width:150px;"/>
         <img src="_static/visible//tsc/T_05.png" alt="tsc T_05" style="width:150px;"/>
         <img src="_static/visible//tsc/T_06.png" alt="tsc T_06" style="width:150px;"/>
         <img src="_static/visible//tsc/T_07.png" alt="tsc T_07" style="width:150px;"/>
     </div>

ttfs
----------------
Overview:
    Time-To-First-Spike (TTFS). It maps input intensity to the time step of the first spike: stronger inputs fire earlier, and weaker inputs fire later. It supports both linear and logarithmic coding modes, and the coding characteristics can be adjusted through parameters such as the threshold and time constant.

Formula:

Linear mode:

.. raw:: html
.. math::
   t^* = \text{clamp}\big(-\tau(\tilde x-1), -\tau(\theta-1)\big) + t_0

Logarithmic mode:

.. raw:: html
.. math::
   t^* = \tau \ln\Big(\frac{\tilde x}{\tilde x-\theta}\Big) + t_0

Where:

- :math:`\theta` is the firing threshold
- :math:`\tau` is the time constant
- :math:`t_0` is the first-spike time offset

References:

    JK Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE'2023.
    https://github.com/jeshraghian/snntorch

.. raw:: html

     <div style="display:flex; gap:8px; align-items:center;">
         <img src="_static/visible//ttfs/T_00.png" alt="ttfs T_00" style="width:150px;"/>
         <img src="_static/visible//ttfs/T_01.png" alt="ttfs T_01" style="width:150px;"/>
         <img src="_static/visible//ttfs/T_02.png" alt="ttfs T_02" style="width:150px;"/>
         <img src="_static/visible//ttfs/T_03.png" alt="ttfs T_03" style="width:150px;"/>
     </div>
     <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
         <img src="_static/visible//ttfs/T_04.png" alt="ttfs T_04" style="width:150px;"/>
         <img src="_static/visible//ttfs/T_05.png" alt="ttfs T_05" style="width:150px;"/>
         <img src="_static/visible//ttfs/T_06.png" alt="ttfs T_06" style="width:150px;"/>
         <img src="_static/visible//ttfs/T_07.png" alt="ttfs T_07" style="width:150px;"/>
     </div>
