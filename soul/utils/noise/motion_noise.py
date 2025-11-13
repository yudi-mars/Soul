import torch

def add_gaussian_noise(x: torch.Tensor, sigma: float):
    '''
    additive gaussian noise for motion sensing inputs

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, C, W)
    sigma : float
        noise standard deviation
    '''
    noise = torch.randn_like(x) * sigma

    return x + noise

def add_drift_noise(x: torch.Tensor, drift_rate: float, dt: float = 1.0):
    '''
    Simulate long-term sensor bias drift or cumulative error

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, C, W)
    drift_rate : float
        Drift rate
    dt : float, optional
        Time length of each time window, by default 1.0
    '''
    T, B, C, W = x.shape
    for t in range(T):
        drift = torch.linspace(0, drift_rate * dt * (W-1), W, device=x.device)
        drift = drift.unsqueeze(0).unsqueeze(0).repeat(B, C, 1)

        x[t] += drift

    return x

def add_impulse_noise(x: torch.Tensor, p: float, amplitude: float=2.0):
    '''
    Sudden step/change or impact noise

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, C, W)
    p : float
        Probability of each element (channel–time cell) being perturbed
    amplitude : float
        Step/change amplitude, by default 2.0
    '''
    mask = torch.rand_like(x) < p
    sign = torch.where(torch.rand_like(x) < 0.5, 1.0, -1.0)
    perturb = mask * sign * amplitude

    return x + perturb

def add_dropouts_noise(x: torch.Tensor, drop_rate: float, fill_value: float = 0.0):
    '''
    Simulate sample loss / missing samples within a time window

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, C, W)
    drop_rate : float
        Probability of loss for each point in the time window (or each channel–time cell)
    fill_value : float, optional
        Fill value for missing samples, by default 0.0
    '''
    mask = torch.randn_like(x) >= drop_rate

    return x * mask + (~mask) * fill_value



