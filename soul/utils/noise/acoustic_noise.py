import random

import torch

def add_gaussian_noise(x: torch.Tensor, sigma: float):
    '''
    Additvie white (gaussian) noise for acoustic sensing inputs (Mel-frequency spectrogram)

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, W, C)
    sigma : float
        noise standard deviation
    '''
    for t in range(x.shape[0]):
        noise = torch.randn_like(x[t]) * sigma
        x[t] += noise

    return x

def add_impulse_noise(x: torch.Tensor, prob: float, amplitude: float=2.):
    '''
    Simulate burst impulse (click/pop) interference by randomly applying strong perturbations to certain frames or channels in the spectrogram

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, W, C)
    prob : float
        The probability of each element being perturbed
    amplitude : float
        Perturbation amplitude (can be positive or negative, or an additive/subtractive value), by default 2.0
    '''
    for t in range(x.shape[0]):
        mask = torch.rand_like(x[t]) < prob
        # random positive/negative pertubation
        sign = torch.where(torch.rand_like(x[t]) < 0.5, 1.0, -1.0)
        perturb = mask * sign * amplitude

        x[t] += perturb

    return x

def add_dropouts_noise(
        x: torch.Tensor,
        drop_ratio: float = 0.0,
        freq_drop_ratio: float = 0.0,
        num_time_masks: int = 1,
        num_freq_masks: int = 1,
        fill_value: float = 0.0):
    '''
    Apply time-axis and/or frequency-axis masking to the input mel-spectrogram

    Parameters
    ----------
    x : torch.Tensor
        tensor shape (B, W, C)
    drop_ratio : int, optional
        Maximum proportion of time-axis masking (in frames), by default 0
    freq_drop_ratio : int, optional
        Maximum proportion of frequency-axis masking (in channels), by default 0
    num_time_masks : int, optional
        Number of time-axis masks applied per sample, by default 1
    num_freq_masks : int, optional
        Number of frequency-axis masks applied per sample, by default 1
    fill_value : float, optional
        Mask fill value, by default 0.0
    '''
    T, B, W, C = x.shape
    for t in range(T):
        out = x[t].clone()
        
        # time window dropout
        if drop_ratio > 0:
            max_w = int(W * drop_ratio)
            for _ in range(num_time_masks):
                w = random.randint(0, max_w)
                w0 = random.randint(0, max(0, W - w))
                out[:, w0:w0 + w, :] = fill_value

        # frequency channel dropout
        if freq_drop_ratio > 0:
            max_f = int(C * freq_drop_ratio)
            for _ in range(num_freq_masks):
                f = random.randint(0, max_f)
                f0 = random.randint(0, max(0, C - f))
                out[:, :, f0:f0 + f] = fill_value

        x[t] = out

    return x


if __name__ == '__main__':
    batch = torch.randn(2, 1, 10, 5) # (T, B, W, C)
    print(batch)
    
    print('=' * 6 + 'dropout noise' + '=' * 6)
    batch_aug = add_dropouts_noise(batch.clone(), drop_ratio=0.3)
    print(batch_aug)

    # print('=' * 6 + 'impulse noise' + '=' * 6)
    # batch_aug = add_impulse_noise(batch.clone(), prob=0.5, amplitude=2.)
    # print(batch_aug)

    print('=' * 6 + 'gaussian noise' + '=' * 6)
    batch_aug = add_gaussian_noise(batch.clone(), sigma=0.1)
    print(batch_aug)