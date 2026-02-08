编码方式
========

本章介绍 Soul 工具包支持的编码方式。编码器的任务是把外部输入（图像、音频、传感器、时间序列等）映射为 “时间优先（time-first）” 的序列形式，以便后续 SNN 在离散时间步上处理。

**统一约定**
- 设输入张量为 :math:`\mathbf{x}`，编码长度为 :math:`T=\texttt{num_steps}`。
- 编码器内部会先把输入转为 ``float32`` 并做 min-max 归一化到 :math:`[0,1]`：记为 :math:`\tilde{\mathbf{x}}\in[0,1]`。
- 输出采用时间优先维度顺序：输出形状为 ``(T, *inputs.shape)``。

bsa
----------------
介绍：
    具有形状感知路由的本氏脉冲算法（Ben's Spiker Algorithm, BSA）。BSA 更像是一类“把连续信号稀疏化为脉冲列，同时可用固定核重建”的匹配追踪过程：通过一个因果核（causal kernel）去拟合输入，残差超过阈值时发放脉冲并更新残差。它适合音频/传感器等连续波形（也可对每个通道独立处理图像/特征图）。

2D输入模式公式：

.. math::

    s(t)=\mathbb{1}\big[e(t)>\theta\big],\quad e(t)=x(t)-\sum_{\tau<t} s(\tau)\,h(t-\tau)

其中 :math:`h` 为指数衰减核，:math:`\theta` 为自适应阈值，:math:`e(t)` 为当前残差。

实现补充（2D模式）：令

.. math::

    \mathrm{corr}(w,c)=\sum_{\tau=0}^{L-1}\,\tilde x_{w+\tau,c}\,h(\tau),\quad
    a=\mathrm{clamp}\Big(\frac{\mathrm{corr}-\theta}{c_{\max}-\theta},0,1\Big),
    \;t^*=\left\lfloor (T-1)(1-a)\right\rfloor.

3D输入模式公式：

.. math::

    t^* = \left\lfloor\frac{\mathrm{rank}(\tilde x)}{P-1}(T-1)\right\rfloor,\quad s_t=\mathbb{1}[t=t^*]

其中 :math:`\mathrm{rank}(\tilde x)\in\{0,\dots,P-1\}` 是降序排序的位置编号（最大值 rank=0），:math:`P=H\times W` 是空间位置总数。

实现备注：

- 2D输入使用自适应阈值调整，目标脉冲密度范围为[0.10, 0.35] 
- 3D输入实际采用排序编码策略
引用：

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
介绍：
    爆发式编码（Burst coding）。它将输入强度映射为一段短脉冲包：输入越强，同一窗口内的脉冲越密集（ISI 越小）。每个元素最多发放B_max个脉冲，脉冲间隔由输入强度决定。支持2D和3D输入，输出维度与输入相同。

公式：

.. math::

    B_{\max} = \min(4,T),\quad
    \mathrm{ISI}_{\max} = \max\left(1, \frac{T-1}{B_{\max}-1}\right),

.. math::

    B(\tilde x)=1+\left\lfloor\tilde x\,(B_{\max}-1)\right\rfloor,
    \;\mathrm{ISI}(\tilde x)=\mathrm{round}\big(\mathrm{ISI}_{\max}-(\mathrm{ISI}_{\max}-1)\,\tilde x\big),

.. math::

    t_k=k\cdot\mathrm{ISI}(\tilde x),\quad s(t)=\sum_{k=0}^{B(\tilde x)-1} \mathbb{1}[t=t_k].

其中：

- :math:`\tilde x` 是归一化到[0,1]的输入值
- :math:`B_{\max}` 是最大脉冲数（不超过4）
- :math:`\mathrm{ISI}_{\max}` 是最大脉冲间隔
- :math:`B(\tilde x)` 是实际脉冲数
- :math:`\mathrm{ISI}(\tilde x)` 是实际脉冲间隔
实现备注：

- ISI限制为至少1，确保脉冲不会重叠
- 脉冲时间通过累积方式计算：t_k = k * ISI
引用：

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
介绍：
    直接编码（Direct）。把归一化后的输入复制T份直接作为每个时间步的驱动信号。

公式：

.. math::

   u(t)=\tilde x,\quad \forall t\in\{0,\dots,T-1\}

引用：

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
介绍：
    二进制相位编码（Phase coding）。它将归一化到[0,1]的输入值扩展为8位二进制分数，并将每一位映射到一个相位（时间步）上的脉冲。前8个时间步按权重(1, 1/2, 1/4, ...)递减，超过8步则循环重复前8位的相位模式。
公式：

.. math::
    q = \left\lfloor\tilde x\cdot 256\right\rfloor,\quad
    s_t = \left((q \gg (8-t))\;\bmod\;2\right),\quad
    o_t = s_t\;2^{-(t-1)}\quad (t=1,\dots,8)

.. math::
    o_t = o_{t \bmod 8}\quad (t>8)

其中：

- :math:`\tilde x` 是归一化到[0,1]的输入值
- :math:`q` 是量化到8位的整数值
- :math:`s_t` 是第t位的二进制值
- :math:`o_t` 是第t个时间步的输出值（带权重）
实现备注：

- 前8个时间步的权重按2的负幂次递减
- 超过8步时循环重复前8位的模式
引用：

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
介绍：
    群体编码（Population coding）。使用高斯调谐曲线将输入值映射到一组神经元群体响应。每个神经元对特定输入值范围有最大响应，响应强度由高斯函数决定。实现中使用均匀分布在[0,1]上的调谐中心，beta=100.0控制响应的尖锐程度。支持2D和3D输入，输出维度由num_steps决定。

公式：

.. math::

    y_i = \exp\big(-\beta(\tilde x-\mu_i)^2\big),\quad i=1,\dots,N

其中：

- :math:`\mu_i` 是第i个神经元的调谐中心，均匀分布在[0,1]上
- :math:`\beta=100.0` 控制响应的尖锐程度
引用：

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
介绍：
    排序编码（Rank-Order coding）。对每个空间位置（或样本位置）沿通道维度排序，排序等级（rank）映射到离散时间步，然后在该时间步发放一次脉冲。强度越大，rank 越靠前，从而越早发放。

.. math::

    \hat x = \tilde x + \epsilon\frac{i}{N-1},\quad
    t^* = \left\lfloor\frac{\mathrm{rank}(\hat x)}{N-1}(T-1)\right\rfloor,\quad s_t=\mathbb{1}[t=t^*]

其中：

- :math:`\epsilon` 是小常数（代码中为1e-6）
- :math:`i` 是元素索引
- :math:`N` 是总元素数
- :math:`\mathrm{rank}(\hat x)\in\{0,\dots,N-1\}` 是降序排序的位置编号（最大值 rank=0）
实现备注：

- 使用确定性平移处理相等值，避免大量相等值导致的脉冲聚集
- 每个元素恰好发放一次脉冲（one-spike-per-element）

引用：

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
介绍：
    速率编码（Rate coding）又称泊松编码(Possion coding)。它将输入强度映射为单位时间窗内的放电次数/概率（即脉冲率）用单位时间窗内的放电次数/概率来表征输入强度。

公式：

.. math::

   s_t \sim \mathrm{Bernoulli}(p=\tilde x),\quad t=0,\dots,T-1


引用：

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
介绍：
    时间开关编码编码（Temporal Switch Coding）。它强调保留输入的时间结构：若输入本身包含时间轴（例如原始传感器序列、骨骼序列、事件流），TSC 往往先把该时间轴转为 time-first，并在每个时间步上做轻量的幅值到脉冲的映射（例如阈值化或速率采样）。

公式：

.. math::

    p^*=\left\lfloor \tilde p\,(T-1)\right\rfloor+1,
    \quad \text{若 }p^*\ge 2:\; s_{0}=\operatorname{sign}(\tilde p),\; s_{p^*-1}=-\operatorname{sign}(\tilde p),
    \quad \text{否则无脉冲}.

引用：

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
介绍：
    首次脉冲时间编码（Time-To-First-Spike, TTFS）。它将输入强度映射为首次发放的时间步：输入越强，发放越早；输入越弱，发放越晚。支持线性和对数两种编码模式，可通过阈值、时间常数等参数调节编码特性。

公式：

线性模式：

.. raw:: html
.. math::
   t^* = \text{clamp}\big(-\tau(\tilde x-1), -\tau(\theta-1)\big) + t_0

对数模式：

.. raw:: html
.. math::
   t^* = \tau \ln\Big(\frac{\tilde x}{\tilde x-\theta}\Big) + t_0

其中：

- :math:`\theta` 是发放阈值
- :math:`\tau` 是时间常数
- :math:`t_0` 是首次脉冲时间偏移

引用：

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