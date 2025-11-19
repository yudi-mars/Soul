'''
Filename: targets_rate.py
Author: Weisong Zhang <zws1415@zju.edu.cn>
Date Created: 2025-11-11
Description:
    implementation of targets_rate coding mechanism for SNN inputs.

References:
    - JK Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE'2023.
    https://github.com/jeshraghian/snntorch
'''

import torch
dtype = torch.float

def targets_rate_coding(
    targets,
    num_steps=False,
    first_spike_time=0,
    correct_rate=1,
    incorrect_rate=0,
    on_target=1,
    off_target=0,
    firing_pattern="regular",
    interpolate=False,
    epsilon=1e-7,
):

    """Spike rate encoding of targets. Input tensor must be one-dimensional
    with target indexes.
    If the output tensor is time-varying, the returned tensor will have
    time in the first dimension.
    If it is not time-varying, then the returned tensor will omit the time
    dimension and use batch first.
    If ``first_spike_time!=0``, ``correct_rate!=1``, or ``incorrect_rate!=0``,
    the output tensor will be time-varying.

    If ``on_target=1``, ``off_target=0``, and ``interpolate=False``,
    then the target may sensibly be applied as a target for the output spike.
    IF any of the above 3 conditions do not hold, then the target would
    be better suited for the output membrane potential.


    Example::

        a = torch.Tensor([4])

        # one-hot
        spikegen.targets_rate(a, num_classes=5)
        >>> (tensor([[0., 0., 0., 0., 1.]]), )

        # first spike time delay, spike evolution over time
        spikegen.targets_rate(a, num_classes=5, num_steps=5,
        first_spike_time=2).size()
        >>> torch.Size([5, 1, 5])
        spikegen.targets_rate(a, num_classes=5, num_steps=5,
        first_spike_time=2)[:, 0, 4]
        >>> (tensor([0., 0., 1., 1., 1.]))

        # note: time has not been repeated because every time step
         would be identical where first_spike_time defaults to 0
        spikegen.targets_rate(a, num_classes=5, num_steps=5).size()
        >>> torch.Size([1, 5])

        # on/off targets - membrane evolution over time
        spikegen.targets_rate(a, num_classes=5, num_steps=5,
        first_spike_time=2, on_target=1.2, off_target=0.5)[:, 0, 4]
        >>> (tensor([0.5000, 0.5000, 1.2000, 1.2000, 1.2000]))

        # correct rate at 25% + linear interpolation of membrane evolution
        spikegen.targets_rate(a, num_classes=5, num_steps=5,
        correct_rate=0.25, on_target=1.2,
        off_target=0.5, interpolate=True)[:, 0, 4]
        >>> tensor([1.2000, 0.5000, 0.7333, 0.9667, 1.2000])


    :param targets: Target tensor for a single batch. The target
        should be a class index in the range [0, C-1]
        where C=number of classes.
    :type targets: torch.Tensor

    :param num_classes: Number of outputs.
    :type num_classes: int

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param first_spike_time: Time step for first spike to occur,
        defaults to ``0``
    :type first_spike_time: int, optional

    :param correct_rate: Firing frequency of correct class as a
        ratio, e.g., ``1`` enables firing at every step; ``0.5``
        enables firing at 50% of steps, ``0`` means no firing,
        defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es), e.g.,
        ``1`` enables firing at every step; ``0.5``
        enables firing at 50% of steps,
        ``0`` means no firing, defaults to ``0``
    :type incorrect_rate: float, optional

    :param on_target: Target at spike times, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Target during refractory period, defaults to ``0``
    :type off_target: float, optional

    :param firing_pattern: Firing pattern of correct and incorrect classes.
        ``'regular'`` enables periodic firing, ``'uniform'`` samples spike
        times from a uniform distributions (duplicates are removed),
        ``'poisson'`` samples from a binomial distribution at each step
        where each probability is the firing frequency,
        defaults to ``'regular'``
    :type firing_pattern: string, optional

    :param interpolate: Applies linear interpolation such that there
        is a gradually increasing target
        up to each spike, defaults to ``False``
    :type interpolate: Bool, optional

    :param epsilon:  A tiny positive value to avoid rounding errors when
        using torch.arange, defaults to ``1e-7``
    :type epsilon: float, optional

    :return: rate coded target of output neurons. If targets are
        time-varying, the output tensor will use time-first dimensions.
        Otherwise, time is omitted from the tensor.
    :rtype: torch.Tensor

    """

    if not 0 <= correct_rate <= 1 or not 0 <= incorrect_rate <= 1:
        raise Exception(
            f"``correct_rate``{correct_rate} and "
            f"``incorrect_rate``{incorrect_rate} must be between 0 and 1."
        )

    if not num_steps and (correct_rate != 1 or incorrect_rate != 0):
        raise Exception(
            "``num_steps`` must be passed if correct_rate is not 1 or "
            "incorrect_rate is not 0."
        )

    if incorrect_rate > correct_rate:
        raise Exception(
            "``correct_rate`` must be greater than ``incorrect_rate``."
        )

    if firing_pattern.lower() not in ["regular", "uniform", "poisson"]:
        raise Exception(
            "``firing_pattern`` must be either 'regular', 'uniform' or "
            "'poisson'."
        )

    device = targets.device
    num_classes = targets.shape[1] if targets.ndim == 2 else None
    # return a non time-varying tensor
    if correct_rate == 1 and incorrect_rate == 0:
        if first_spike_time == 0:
            if on_target > off_target:
                return torch.clamp(
                    to_one_hot(targets, num_classes) * on_target, off_target
                )
            else:
                return (
                    to_one_hot(targets, num_classes) * on_target
                    + ~(to_one_hot(targets, num_classes)).bool() * off_target
                )

        # return time-varying tensor: off up to first_spike_time,
        # then correct classes are on after
        if first_spike_time > 0:
            spike_targets = torch.clamp(
                to_one_hot(targets, num_classes) * on_target, off_target
            )
            spike_targets = spike_targets.repeat(
                tuple(
                    [num_steps]
                    + torch.ones(len(spike_targets.size()), dtype=int).tolist()
                )
            )
            spike_targets[0:first_spike_time] = off_target
            return spike_targets

            # executes if on/off firing rates are not 100% / 0%
    else:
        one_hot_targets = to_one_hot(targets, num_classes)
        one_hot_inverse = to_one_hot_inverse(one_hot_targets)

        # project one-hot-encodings along the time-axis (0th dim)
        one_hot_targets = one_hot_targets.repeat(
            tuple(
                [num_steps]
                + torch.ones(len(one_hot_targets.size()), dtype=int).tolist()
            )
        )
        one_hot_inverse = one_hot_inverse.repeat(
            tuple(
                [num_steps]
                + torch.ones(len(one_hot_inverse.size()), dtype=int).tolist()
            )
        )

        # create tensor of spike_targets for correct class
        correct_spike_targets, correct_spike_times = target_rate_code(
            num_steps=num_steps,
            first_spike_time=first_spike_time,
            rate=correct_rate,
            firing_pattern=firing_pattern,
        )
        correct_spikes_one_hot = one_hot_targets * correct_spike_targets.to(
            device
        ).unsqueeze(-1).unsqueeze(
            -1
        )  # the two unsquezes make the dims of correct_spikes
        # num_steps x 1 x 1, s.t. time is broadcast in every other direction

        # create tensor of spike targets for incorrect class
        incorrect_spike_targets, incorrect_spike_times = target_rate_code(
            num_steps=num_steps,
            first_spike_time=first_spike_time,
            rate=incorrect_rate,
            firing_pattern=firing_pattern,
        )
        incorrect_spikes_one_hot = (
            (one_hot_inverse * incorrect_spike_targets)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # the two unsquezes make the dims of correct_spikes
        # num_steps x 1 x 1, s.t. time is broadcasted in every other direction

        # merge the incorrect and correct tensors
        if not interpolate:
            return torch.clamp(
                (
                    incorrect_spikes_one_hot.to(device)
                    + correct_spikes_one_hot.to(device)
                )
                * on_target,
                off_target,
            )

        # interpolate values between spikes
        else:
            correct_spike_targets = one_hot_targets * (
                rate_interpolate(
                    correct_spike_times,
                    num_steps=num_steps,
                    on_target=on_target,
                    off_target=off_target,
                    epsilon=epsilon,
                )
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )  # the two unsquezes make the dims of correct_spikes
            # num_steps x 1 x 1, s.t. the time is broadcasted in every
            # other direction
            incorrect_spike_targets = one_hot_inverse * (
                rate_interpolate(
                    incorrect_spike_times,
                    num_steps=num_steps,
                    on_target=on_target,
                    off_target=off_target,
                    epsilon=epsilon,
                )
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            return correct_spike_targets + incorrect_spike_targets

def target_rate_code(
    num_steps, first_spike_time=0, rate=1, firing_pattern="regular"
):
    """
    Rate coding a single output neuron of tensor of length ``num_steps``
    containing spikes, and another tensor containing the spike times.


    Example::

        spikegen.target_rate_code(num_steps=5, rate=1)
        >>> (tensor([1., 1., 1., 1., 1.]), tensor([0, 1, 2, 3, 4]))

        spikegen.target_rate_code(num_steps=5, first_spike_time=3, rate=1)
        >>> (tensor([0., 0., 0., 1., 1.]), tensor([3, 4]))

        spikegen.target_rate_code(num_steps=5, rate=0.3)
        >>> (tensor([1., 0., 0., 1., 0.]), tensor([0, 3]))

        spikegen.target_rate_code(
        num_steps=5, rate=0.3, firing_pattern="poisson")
        >>> (tensor([0., 1., 0., 1., 0.]), tensor([1, 3]))

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param first_spike_time: Time step for first spike to occur,
        defaults to ``0``
    :type first_spike_time: int, optional

    :param rate: Firing frequency as a ratio, e.g., ``1``
        enables firing at every step; ``0.5`` enables firing at 50% of steps,
        ``0`` means no firing, defaults to ``1``
    :type rate: float, optional

    :param firing_pattern: Firing pattern of correct and
        incorrect classes. ``'regular'`` enables periodic firing,
        ``'uniform'`` samples spike times from a uniform distributions
        (duplicates are removed), ``'poisson'`` samples from a binomial
        distribution at each step where each probability
        is the firing frequency,
        defaults to ``'regular'``
    :type firing_pattern: string, optional

    :return: rate coded target of single neuron class of length ``num_steps``
    :rtype: torch.Tensor

    :return: rate coded spike times in terms of steps
    :rtype: torch.Tensor

    """

    if not 0 <= rate <= 1:
        raise Exception(f"``rate``{rate} must be between 0 and 1.")

    if first_spike_time > num_steps:
        raise Exception(
            f"``first_spike_time {first_spike_time} must be less "
            f"than num_steps {num_steps}."
        )

    if rate == 0:
        return torch.zeros(num_steps), torch.Tensor()

    if firing_pattern.lower() == "regular":
        spike_times = torch.arange(first_spike_time, num_steps, 1 / rate)
        return (
            torch.zeros(num_steps).scatter(0, spike_times.long(), 1),
            spike_times.long(),
        )

    elif firing_pattern.lower() == "uniform":
        spike_times = (
            torch.rand(
                len(torch.arange(first_spike_time, num_steps, 1 / rate))
            )
            * (num_steps - first_spike_time)
            + first_spike_time
        )
        return (
            torch.zeros(num_steps).scatter(0, spike_times.long(), 1),
            spike_times.long(),
        )

    elif firing_pattern.lower() == "poisson":
        spike_targets = torch.bernoulli(
            torch.cat(
                (
                    # torch.zeros((first_spike_time), device=device),
                    # torch.ones((num_steps - first_spike_time),
                    # device=device) * rate,
                    torch.zeros((first_spike_time)),
                    torch.ones((num_steps - first_spike_time)) * rate,
                )
            )
        )
        return spike_targets, torch.where(spike_targets == 1)[0]
    
def to_one_hot_inverse(one_hot_targets):
    """Boolean inversion of a matrix of 1's and 0's.
    Used to merge the targets of correct and incorrect neuron classes in
    ``targets_rate``.

    Example::

        a = torch.Tensor([0, 0, 0, 0, 1])
        spikegen.to_one_hot_inverse(a)
        >>> tensor([[1., 1., 1., 1., 0.]])

    """

    one_hot_inverse = one_hot_targets.clone()
    one_hot_inverse[one_hot_targets == 0] = 1
    one_hot_inverse[one_hot_targets != 0] = 0

    return one_hot_inverse

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

def from_one_hot(one_hot_label):
    """Convert one-hot encoding back into an integer

    Example::

        one_hot_label = torch.tensor([[1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]])
        spikegen.from_one_hot(one_hot_label)
        >>> tensor([0, 1, 2, 3])

    :param targets: one-hot label vector
    :type targets: torch.Tensor

    :return: targets
    :rtype: torch.Tensor

    """

    # one_hot_label = torch.where(one_hot_label == 1)[0][0]
    # return int(one_hot_label)

    one_hot_label = torch.where(one_hot_label == 1)[0]
    return one_hot_label

def rate_interpolate(
    spike_time, num_steps, on_target=1, off_target=0, epsilon=1e-7
):
    """Apply linear interpolation to a tensor of target spike times to
    enable gradual increasing membrane.

    Example::

        a = torch.Tensor([0, 4])
        spikegen.rate_interpolate(a, num_steps=5)
        >>> tensor([1.0000, 0.0000, 0.3333, 0.6667, 1.0000])

        spikegen.rate_interpolate(a, num_steps=5, on_target=1.25,
        off_target=0.25)
        >>> tensor([1.2500, 0.2500, 0.5833, 0.9167, 1.2500])

    :param spike_time: spike time targets in terms of steps
    :type targets: torch.Tensor

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param on_target: Target at spike times, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Target during refractory period, defaults to ``0``
    :type off_target: float, optional

    :param epsilon:  A tiny positive value to avoid rounding errors when
        using torch.arange, defaults to ``1e-7``
    :type epsilon: float, optional

    :return: interpolated target of output neurons. Output tensor will
        use time-first dimensions.
    :rtype: torch.Tensor

    """

    # if no spikes
    if not spike_time.numel():
        return torch.ones((num_steps)) * off_target

    current_time = -1

    interpolated_targets = torch.Tensor([])

    for step in range(num_steps):
        if step in spike_time:
            if step == (current_time + 1):
                interpolated_targets = torch.cat(
                    (interpolated_targets, torch.Tensor([on_target]))
                )
            else:
                interpolated_targets = torch.cat(
                    (
                        interpolated_targets,
                        torch.arange(
                            off_target,
                            on_target + epsilon,
                            (on_target - off_target)
                            / (step - current_time - 1),
                        ),
                    )
                )
            current_time = step

    if torch.max(spike_time) < num_steps - 1:
        for step in range(int(torch.max(spike_time).item()), num_steps - 1):
            interpolated_targets = torch.cat(
                (interpolated_targets, torch.Tensor([off_target]))
            )
    return interpolated_targets

