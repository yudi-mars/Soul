import torch

dtype = torch.float

__all__ = ['rate_coding', 'temporal_coding', 'direct_coding']

def direct_coding(data, num_steps):
    '''
    direct coding for inputs by repeating roughly

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]

    Returns
    -------
    torch.Tensor
        rate encoding spike train of input features of shape [num_steps x batch x input_size]
    '''
    data = data.unsqueeze(0).repeat(tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist()))

    return data # (T, B, C, H, W)

def rate_conv(data):
    '''
    Convert tensor into Poisson spike trains using the features as
    the mean of a binomial distribution. Values outside the range of 
    [0, 1] are clipped so they can be treated as probabilities.

    Example::

        # 100% chance of spike generation
        a = torch.Tensor([1, 1, 1, 1])
        spikegen.rate_conv(a)
        >>> tensor([1., 1., 1., 1.])

        # 0% chance of spike generation
        b = torch.Tensor([0, 0, 0, 0])
        spikegen.rate_conv(b)
        >>> tensor([0., 0., 0., 0.])

        # 50% chance of spike generation per time step
        c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
        spikegen.rate_conv(c)
        >>> tensor([0., 1., 0., 1.])

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]

    Returns
    -------
    torch.Tensor
        rate encoding spike train of input features of shape [num_steps x batch x input_size]
    '''

    # Clip all features between 0 and 1 so they can be used as probabilities.
    clipped_data = torch.clamp(data, min=0, max=1)
    spike_data = torch.bernoulli(clipped_data)

    return spike_data


def rate_coding(data, num_steps=False, gain=1, offset=0, first_spike_time=0, time_var_input=False):
    '''
    Spike rate encoding of input data. Convert tensor into Poisson spike trains using the 
    features as the mean of a binomial distribution. If `num_steps` is specified, then the 
    data will be first repeated in the first dimension before rate encoding.

    If data is time-varying, tensor dimensions use time first.

    Example::

        # 100% chance of spike generation
        a = torch.Tensor([1, 1, 1, 1])
        spikegen.rate(a, num_steps=1)
        >>> tensor([1., 1., 1., 1.])

        # 0% chance of spike generation
        b = torch.Tensor([0, 0, 0, 0])
        spikegen.rate(b, num_steps=1)
        >>> tensor([0., 0., 0., 0.])

        # 50% chance of spike generation per time step
        c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
        spikegen.rate(c, num_steps=1)
        >>> tensor([0., 1., 0., 1.])

        # Increasing num_steps will increase the length of
        # the first dimension (time-first)
        print(c.size())
        >>> torch.Size([1, 4])

        d = spikegen.rate(torch.Tensor([0.5, 0.5, 0.5, 0.5]), num_steps = 2)
        print(d.size())
        >>> torch.Size([2, 4])

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]
    num_steps : int, optional
        Number of time steps. Only specify if input data does not already have time dimension, by default False
    gain : float, optional
        Scale input features by the gain, by default 1
    offset : torch.optim, optional
        Shift input features by the offset, by default 0
    first_spike_time : int, optional
        Time to first spike, by default 0
    time_var_input : bool, optional
        Set to ``True`` if input tensor is time-varying. Otherwise, `first_spike_time!=0` will modify the wrong dimension., by default False

    Returns
    -------
    torch.Tensor
        rate encoding spike train of input features of shape [num_steps x batch x input_size]
    '''
    if first_spike_time < 0 or num_steps < 0:
        raise ValueError("``first_spike_time`` and ``num_steps`` cannot be negative.")
    
    if first_spike_time > (num_steps - 1):
        if num_steps:
            raise ValueError(f"first_spike_time ({first_spike_time}) must be equal to or less than num_steps-1 ({num_steps-1}).")
        if not time_var_input:
            raise ValueError(
                "If the input data is time-varying, set "
                "``time_var_input=True``.\n If the input data is not "
                "time-varying, ensure ``num_steps > 0``."
            )
        
    if first_spike_time > 0 and not time_var_input and not num_steps:
        raise ValueError(
            "``num_steps`` must be specified if both the input is not "
            "time-varying and ``first_spike_time`` is greater than 0."
        )

    if time_var_input and num_steps:
        raise Exception(
            "``num_steps`` should not be specified if input is "
            "time-varying, i.e., ``time_var_input=True``.\n "
            "The first dimension of the input data + ``first_spike_time`` "
            "will determine ``num_steps``."
        )
    
    device = data.device
    # for time-varying input data
    if time_var_input:
        spike_data = rate_conv(data)

        # zeros are added directly to the start of 0th (time) dimension
        if first_spike_time > 0:
            spike_data = torch.cat((
                torch.zeros(
                    tuple([first_spike_time] + list(spike_data[0].size())),
                    device=device,
                    dtype=dtype),
                    spike_data,
            ))
    # intended for time-static input data
    else:
        time_data = (
            data.repeat(
                tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist())
            ) * gain + offset
        )

        spike_data = rate_conv(time_data)

        # zeros are multiplied by the start of the 0th (time) dimension
        if first_spike_time > 0:
            spike_data[0:first_spike_time] = 0

    return spike_data

def latency_code(
    data,
    num_steps=False,
    threshold=0.01,
    tau=1,
    first_spike_time=0,
    normalize=False,
    linear=False,
    epsilon=1e-7,
):
    '''
    Latency encoding of input data. Convert input features or
    target labels to spike times. Assumes a LIF neuron model
    that charges up with time constant tau by default.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency_code(a, num_steps=5, normalize=True, linear=True)
        >>> (tensor([3.9200, 2.0000, 0.0000]), tensor([False, False, False]))

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]
    num_steps : int, optional
        Number of time steps. Explicitly needed if ``normalize=True``, by default False (then changed to ``1`` if ``normalize=False``)
    threshold : float, optional
        Input features below the threhold will fire at the final time step unless ``clip=True`` in which case they will not fire at all, by default 0.01
    tau : float, optional
        RC Time constant for LIF model used to calculate firing time, by default 1
    first_spike_time : int, optional
        first_spike_time: Time to first spike, by default 0
    normalize : bool, optional
        Option to normalize the latency code such that the final spike(s) occur within num_steps, by default False
    linear : bool, optional
        Apply a linear latency code rather than the default logarithmic code, by default False
    epsilon : float, optional
        A tiny positive value to avoid rounding errors when using torch.arange, by default 1e-7

    Returns
    -------
    torch.Tensor
        latency encoding spike times of features
    '''
    idx = data < threshold

    if not linear:
        spike_time = latency_code_log(
            data,
            num_steps=num_steps,
            threshold=threshold,
            tau=tau,
            first_spike_time=first_spike_time,
            normalize=normalize,
            epsilon=epsilon,
        )

    elif linear:
        spike_time = latency_code_linear(
            data,
            num_steps=num_steps,
            threshold=threshold,
            tau=tau,
            first_spike_time=first_spike_time,
            normalize=normalize,
        )

    return spike_time, idx

def _latency_errors(
    data, num_steps, threshold, tau, first_spike_time, normalize
):
    '''  
    Catch errors for spike time encoding latency functions ``latency_code_linear`` and ``latency_code_log``
    '''

    if (
        threshold <= 0 or threshold >= 1
    ):  # double check if this can just be threshold < 0 instead.
        raise Exception("Threshold must be between 0 and 1.")

    if tau <= 0:  # double check if this can just be threshold < 0 instead.
        raise Exception("``tau`` must be greater than 0.")

    if first_spike_time and num_steps and first_spike_time > (num_steps - 1):
        raise Exception(
            f"first_spike_time ({first_spike_time}) must be equal to "
            f"or less than num_steps-1 ({num_steps-1})."
        )

    # this condition is more broadly caught in latency code by ensuring 0
    # < data < 1
    if first_spike_time and torch.max(data) > 1 and torch.min(data) < 0:
        raise Exception(
            "`first_spike_time` can only be applied to data between "
            "`0` and `1`."
        )

    if first_spike_time < 0:
        raise Exception(
            "``first_spike_time`` [{first_spike_time}] cannot be negative."
        )

    if num_steps < 0:
        raise Exception("``num_steps`` [{num_steps}] cannot be negative.")

    if normalize and not num_steps:
        raise Exception(
            "`num_steps` should not be empty if normalize is set to True."
        )

def latency_code_linear(
    data,
    num_steps=False,
    threshold=0.01,
    tau=1,
    first_spike_time=0,
    normalize=False,
):
    '''
    Linear latency encoding of input data. Convert input features
    or target labels to spike times.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency_code(a, num_steps=5, normalize=True, linear=True)
        >>> (tensor([3.9200, 2.0000, 0.0000]), tensor([False, False, False]))

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]
    num_steps : int, optional
        Number of time steps. Explicitly needed if ``normalize=True``, by default False
    threshold : float, optional
        Input features below the threhold will fire at the final time step, by default 0.01
    tau : float, optional
        Linear time constant used to calculate firing time, by default 1
    first_spike_time : int, optional
        Time to first spike, by default 0
    normalize : bool, optional
        Option to normalize the latency code such that the final spike(s) occur within num_steps, by default False

    Returns
    -------
    torch.Tensor
        linear latency encoding spike times of features
    '''

    _latency_errors(
        data, num_steps, threshold, tau, first_spike_time, normalize
    )  # error checks

    if normalize:
        tau = num_steps - 1 - first_spike_time

    spike_time = (
        torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))
    ) + first_spike_time

    # the following code is intended for negative input data.
    # it is more broadly caught in latency code by ensuring 0 < data < 1.
    # Consider disabling ~(0<data<1) input.
    if torch.min(spike_time) < 0 and normalize:
        spike_time = (
            (spike_time - torch.min(spike_time))
            * (1 / (torch.max(spike_time) - torch.min(spike_time)))
            * (num_steps - 1)
        )
    return spike_time


def latency_code_log(
    data,
    num_steps=False,
    threshold=0.01,
    tau=1,
    first_spike_time=0,
    normalize=False,
    epsilon=1e-7,
):
    '''
    Logarithmic latency encoding of input data. Convert input features
    or target labels to spike times.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency_code(a, num_steps=5, normalize=True)
        >>> (tensor([4.0000, 0.1166, 0.0580]), tensor([False, False, False]))

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]
    num_steps : int, optional
        Number of time steps. Explicitly needed if ``normalize=True``, by default False (then changed to ``1`` if ``normalize=False``)
    threshold : float, optional
        Input features below the threhold will fire at the final time step, by default 0.01
    tau : float, optional
        Logarithmic time constant used to calculate firing time, by default 1
    first_spike_time : int, optional
        Time to first spike, defaults to ``0``, by default 0
    normalize : bool, optional
        Option to normalize the latency code such that the final spike(s) occur within num_steps, by default False
    epsilon : float, optional
        A tiny positive value to avoid rounding errors when using torch.arange, by default 1e-7

    Returns
    -------
    torch.Tensor
        logarithmic latency encoding spike times of features
    '''

    _latency_errors(
        data, num_steps, threshold, tau, first_spike_time, normalize
    )  # error checks

    data = torch.clamp(
        data, threshold + epsilon
    )  # saturates all values below threshold.

    spike_time = tau * torch.log(data / (data - threshold))

    if first_spike_time > 0:
        spike_time += first_spike_time

    if normalize:
        spike_time = (spike_time - first_spike_time) * (
            num_steps - first_spike_time - 1
        ) / torch.max(spike_time - first_spike_time) + first_spike_time

    return spike_time

def latency_interpolate(spike_time, num_steps, on_target=1, off_target=0):
    '''
    Apply linear interpolation to a tensor of target spike times to
    enable gradual increasing membrane. Each spike is assumed to occur 
    from a separate neuron.

    Example::

        a = torch.Tensor([0, 4])
        spikegen.latency_interpolate(a, num_steps=5)
        >>> tensor([[1.0000, 0.0000],
                    [0.0000, 0.2500],
                    [0.0000, 0.5000],
                    [0.0000, 0.7500],
                    [0.0000, 1.0000]])

        spikegen.latency_interpolate(a, num_steps=5, on_target=1.25,
        off_target=0.25)
        >>> tensor([[1.2500, 0.2500],
                    [0.2500, 0.5000],
                    [0.2500, 0.7500],
                    [0.2500, 1.0000],
                    [0.2500, 1.2500]])

    Parameters
    ----------
    spike_time : torch.Tensor
        spike time targets in terms of steps
    num_steps : int
        Number of time steps
    on_target : float, optional
        Target at spike times, by default 1
    off_target : float, optional
        Target during refractory period, by default 0

    Returns
    -------
    torch.Tensor
        interpolated target of output neurons. Output tensor will use time-first dimensions.
    '''
    if on_target < off_target:
        raise Exception(
            f"``on_target`` [{on_target}] must be greater than "
            f"``off_target`` [{off_target}]."
        )

    device = spike_time.device

    spike_time = torch.round(
        spike_time
    ).float()  # Needs to be float as 0s and out-of-bounds spikes are set to 0.5

    spike_time[
        spike_time > num_steps
    ] = 0.5  # avoid div by 0. instead setting spike time to < 1
    # --> (step/spike_time) > 1, which gets clipped.

    interpolated_targets = torch.ones(
        (tuple([num_steps] + list(spike_time.size()))),
        dtype=dtype,
        device=device,
    )

    # offset skips first step if a 0 spike occurs. must be handled
    # separately to avoid div by zero.
    offset = 0
    # index into first step
    if 0 in spike_time:
        interpolated_targets[0] = torch.where(
            spike_time == 0,
            interpolated_targets[0],
            interpolated_targets[0] * 0,
        )  # replace 0's with ones for first spike time, others with 0s
        spike_time[spike_time == 0] = 0.5
        offset = 1

    # i.e., when step/spike_time=1
    for step in range(num_steps - offset):
        interpolated_targets[step + offset] = (step + offset) / spike_time

    # next we clamp those that exceed 1, and rescale
    interpolated_targets = (
        interpolated_targets * (on_target - off_target) + off_target
    )
    interpolated_targets[interpolated_targets > on_target] = off_target

    return interpolated_targets


def temporal_coding(
    data, 
    num_steps=False, 
    threshold=0.01, 
    tau=1, 
    first_spike_time=0, 
    on_target=1,
    off_target=0,
    clip=False,
    normalize=True,
    linear=False,
    interpolate=False,
    bypass=False,
    epsilon=1e-7,
):
    '''
    Temporal (Latency) encoding of input or target label data. Use input features
    to determine time-to-first spike. Expected inputs should be between 0 and 1.

    Assume a LIF neuron model that charges up with time constant tau.
    Tensor dimensions use time first.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency(a, num_steps=5, normalize=True, linear=True)
        >>> tensor([[0., 0., 1.],
                    [0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.],
                    [1., 0., 0.]])

    Parameters
    ----------
    data : torch.Tensor
        Data tensor for a single batch of shape [batch x input_size]
    num_steps : int, optional
        Number of time steps. Explicitly needed if ``normalize=True``, 
        by default ``False`` (then changed to ``1`` if ``normalize=False``)
    threshold : float, optional
        Input features below the threhold will fire at the final time step 
        unless ``clip=True`` in which case they will not fire at all, by default 0.01
    tau : float, optional
        RC Time constant for LIF model used to calculate firing time, by default 1
    first_spike_time : int, optional
        Time to first spike, by default 0
    on_target : float, optional
        Target at spike times, by default 1
    off_target : float, optional
        Target during refractory period, by default 0
    clip : bool, optional
        Option to remove spikes from features that fall
        below the threshold, by default False
    normalize : bool, optional
        Option to normalize the latency code such that the final spike(s) occur 
        within num_steps, by default True
    linear : bool, optional
        Apply a linear latency code rather than the default logarithmic code, by default False
    interpolate : bool, optional
        Applies linear interpolation such that there is a gradually increasing 
        target up to each spike, by default False
    bypass : bool, optional
        Used to block error messages that occur from either: i) spike times exceeding 
        the bounds of ``num_steps``, or ii) if ``num_steps`` is not specified, setting
        ``bypass=True`` allows the largest spike time to set ``num_steps``, by default False
    epsilon : float, optional
        A tiny positive value to avoid rounding errors when using torch.arange, by default 1e-7

    Returns
    -------
    torch.Tensor
        latency encoding spike train of features or labels
    '''
    if torch.min(data) < 0 or torch.max(data) > 1:
        raise ValueError(
            f"Elements of ``data`` must be between [0, 1], but input "
            f"is [{torch.min(data)}, {torch.max(data)}]"
        )
    
    if threshold < 0 or threshold > 1:
        raise Exception(f"``threshold`` [{threshold}] must be between [0, 1]")
    
    if not num_steps and not bypass:
        raise Exception(
            "``num_steps`` must be specified. Alternatively, setting "
            "``bypass=True`` will automatically set ``num_steps`` "
            "to the last spike time. This may lead to uneven tensor "
            "sizes when used in a loop."
        )

    device = data.device

    spike_time, idx = latency_code(
        data,
        num_steps=num_steps,
        threshold=threshold,
        tau=tau,
        first_spike_time=first_spike_time,
        normalize=normalize,
        linear=linear,
        epsilon=epsilon,
    )

    # automatically set num_steps using max element in spike_time
    if not num_steps and bypass:
        num_steps = int(torch.round(torch.max(spike_time)).long() + 1)

        if num_steps <= 0:
            raise Exception(
                f"``num_steps`` [{num_steps}] must be positive. "
                f"This can be specifiedInput data should be normalized "
                f"to larger values or ``threshold`` should be set to a "
                f"smaller value."
            )
        
    if torch.round(torch.max(spike_time)).long() > (num_steps - 1) and not bypass:
        raise Exception(
            f"The maximum value in ``spike_time`` "
            f"[{torch.round(torch.max(spike_time)).long()}] is out of "
            f"bounds for ``num_steps`` [{num_steps}-1].\n To bypass "
            f"this error, set ``bypass=True``.\n Alternatively, constrain "
            f"``spike_time`` within the range of ``num_steps`` "
            f"by either decreasing ``tau`` or ``setting normalize=True``."
        )

    if not interpolate:
        spike_data = torch.zeros(
            (tuple([num_steps] + list(spike_time.size()))),
            dtype=dtype,
            device=device,
        )

        # use rm_idx to remove spikes beyond the range of num_steps
        rm_idx = torch.round(spike_time).long() > num_steps - 1
        rm_idx = torch.round(spike_time).long() > num_steps - 1
        spike_data = (
            spike_data.scatter(0, torch.round(torch.clamp_max(spike_time, num_steps - 1)).long().unsqueeze(0), 1) * ~rm_idx
        )

        # Use idx to remove spikes below the threshold
        if clip:
            spike_data = spike_data * ~idx  # idx is broadcast in T direction

        return torch.clamp(spike_data * on_target, off_target)
    
    else:
        return latency_interpolate(spike_time, num_steps, on_target=on_target, off_target=off_target)

