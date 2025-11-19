'''
Filename: targets_latency.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-11
Description:
    implementation of targets_latency coding mechanism for SNN inputs.

References:
    - JK Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE'2023.
    https://github.com/jeshraghian/snntorch
'''

import torch
from .temporal import temporal_coding
dtype = torch.float

def to_one_hot(targets, num_classes):
    """One hot encoding of target labels.

    Example::

        targets = torch.tensor([0, 1, 2, 3])
        spikegen.targets_to_spikes(targets, num_classes=4)
        >>> tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    :param targets: Target tensor for a single batch
    :type targets: torch.Tensor

    :param num_classes: Number of classes
    :type num_classes: int

    :return: one-hot encoding of targets of shape [batch x num_classes]
    :rtype: torch.Tensor

    """

    if torch.max(targets > num_classes - 1):
        raise Exception(
            f"target [{torch.max(targets)}] is out of bounds for "
            f"``num_classes`` [{num_classes}]"
        )

    device = targets.device

    # Initialize zeros. E.g, for MNIST: (batch_size, 10).
    one_hot = torch.zeros(
        [len(targets), num_classes], device=device, dtype=dtype
    )

    # Unsqueeze converts dims of [100] to [100, 1]
    one_hot = one_hot.scatter(1, targets.type(torch.int64).unsqueeze(-1), 1)

    return one_hot

def targets_latency_coding(
    targets,
    num_steps=False,
    first_spike_time=0,
    on_target=1,
    off_target=0,
    interpolate=False,
    threshold=0.01,
    tau=1,
    clip=False,
    normalize=False,
    linear=False,
    epsilon=1e-7,
    bypass=False,
):

    """Latency encoding of target labels. Use target labels to determine
    time-to-first spike. Expected input is index of correct class.
    The index is one-hot-encoded before being passed to ``spikegen.latency``.

    Assume a LIF neuron model that charges up with time constant tau.
    Tensor dimensions use time first.

    Example::

        a = torch.Tensor([0, 3])
        spikegen.targets_latency(a, num_classes=4, num_steps=5,
        normalize=True).size()
        >>> torch.Size([5, 2, 4])

        # time evolution of correct neuron class
        spikegen.targets_latency(a, num_classes=4, num_steps=5,
        normalize=True)[:, 0, 0]
        >>> tensor([1., 0., 0., 0., 0.])

        # time evolution of incorrect neuron class
        spikegen.targets_latency(a, num_classes=4, num_steps=5,
        normalize=True)[:, 0, 1]
        >>> tensor([0., 0., 0., 0., 1.])

        # correct class w/interpolation
        spikegen.targets_latency(a, num_classes=4, num_steps=5,
        normalize=True, interpolate=True)[:, 0, 0]
        >>> tensor([1., 0., 0., 0., 0.])

        # incorrect class w/interpolation
        spikegen.targets_latency(a, num_classes=4, num_steps=5,
        normalize=True, interpolate=True)[:, 0, 1]
        >>> tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

    :param targets: Target tensor for a single batch. The target
        should be a class index in the range [0, C-1]
        where C=number of classes.
    :type targets: torch.Tensor

    :param num_classes: Number of outputs.
    :type num_classes: int

    :param num_steps: Number of time steps. Explicitly needed if
        ``normalize=True``, defaults to ``False``
        (then changed to ``1`` if ``normalize=False``)
    :type num_steps: int, optional

    :param first_spike_time: Time to first spike, defaults to ``0``.
    :type first_spike_time: int, optional

    :param on_target: Target at spike times, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Target during refractory period, defaults to ``0``
    :type off_target: float, optional

    :param interpolate: Applies linear interpolation such that there is
        a gradually increasing target up to each spike, defaults to ``False``
    :type interpolate: Bool, optional

    :param threshold: Input features below the threhold will fire at the
        final time step unless ``clip=True`` in which case they will not fire
        at all, defaults to ``0.01``
    :type threshold: float, optional

    :param tau:  RC Time constant for LIF model used to calculate firing
        time, defaults to ``1``
    :type tau: float, optional

    :param clip: Option to remove spikes from features that fall below
        the threshold, defaults to ``False``
    :type clip: Bool, optional

    :param normalize: Option to normalize the latency code such that the
        final spike(s) occur within num_steps, defaults to ``False``
    :type normalize: Bool, optional

    :param linear: Apply a linear latency code rather than the default
        logarithmic code, defaults to ``False``
    :type linear: Bool, optional

    :param bypass: Used to block error messages that occur from either: i)
        spike times exceeding the bounds of ``num_steps``, or ii) if
        ``num_steps`` is not specified, setting ``bypass=True``
        allows the largest spike time to set ``num_steps``.
        Defaults to ``False``
    :type bypass: bool, optional

    :param epsilon: A tiny positive value to avoid rounding errors when
        using torch.arange, defaults to ``1e-7``
    :type epsilon: float, optional

    :return: latency encoding spike train of features or labels
    :rtype: torch.Tensor

    """

    if targets.dtype in (torch.long, torch.int64, torch.int32):
        num_classes = int(targets.max().item()) + 1

    elif targets.ndim == 1:
        targets = targets.to(torch.long)
        num_classes = int(targets.max().item()) + 1

    elif targets.ndim == 2:
        num_classes = int(targets.shape[1])
    else:
        raise ValueError(f"targets shape {tuple(targets.shape)} is not supported.")

    return temporal_coding(
        to_one_hot(targets, num_classes),
        num_steps=num_steps,
        first_spike_time=first_spike_time,
        on_target=on_target,
        off_target=off_target,
        interpolate=interpolate,
        threshold=threshold,
        tau=tau,
        clip=clip,
        normalize=normalize,
        linear=linear,
        bypass=bypass,
        epsilon=epsilon,
    )