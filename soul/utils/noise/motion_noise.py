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

def add_dropouts_noise(x: torch.Tensor, drop_ratio: float = 0.1, fill_value: float = 0.0):
    '''
    Simulate sample loss / missing samples within a time window

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, C, W)
    drop_ratio : float
        Probability of loss for each point in the time window (or each channel–time cell)
    fill_value : float, optional
        Fill value for missing samples, by default 0.0
    '''
    time_step, B, C, W = x.shape
    out = x.clone()

    mask = (torch.rand(B, W, device=x.device) >= drop_ratio)
    # mask shape (B, W); broadcast to (B, C, W)
    mask = mask.unsqueeze(1).unsqueeze(0).expand(time_step, B, C, W)
    out = out * mask + (~mask) * fill_value

    return out

if __name__ == '__main__':
    batch = torch.randn(2, 1, 3, 20) # (T, B, C, W)
    print(batch)
    
    print('=' * 6 + 'gaussian noise' + '=' * 6)
    batch_aug = add_gaussian_noise(batch.clone(), sigma=0.1)
    print(batch_aug)

    print('=' * 6 + 'dropouts noise' + '=' * 6)
    batch_aug = add_dropouts_noise(batch.clone(), drop_ratio=0.9)
    print(batch_aug)
