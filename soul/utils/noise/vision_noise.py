import torch

def add_gaussian_noise(inputs: torch.tensor, sigma: float):
    '''
    Additive gaussian noise for vision sensing inputs

    Parameters
    ----------
    inputs : torch.tensor
        input (T, B, C, H, W), float in [0, 1]
    sigma : float
        noise standard deviation
    clip : bool, optional
        clip the tensor to [0, 1], by default True
    '''
    for t in range(inputs.shape[0]):
        noise = torch.randn_like(inputs[t]) * sigma
        inputs[t] += noise

    return inputs

def add_impulse_noise(inputs: torch.Tensor, prob: float):
    '''
    Impulse noise for vision sensing inputs (a.k.a., Salt & Pepper)

    Parameters
    ----------
    inputs : torch.Tensor
        input (T, B, C, H, W), float in [0, 1]
    prob : float
        replacing probability for each pixel to be "salt & pepper"
    clip : bool, optional
        clip the tensor to [0, 1], by default True, by default True
    '''
    for t in range(inputs.shape[0]):
        B, C, H, W = inputs[t].shape

        out = inputs[t].clone() # (B, C, H, W)
        mask = torch.rand(B, 1, H, W, device=inputs.device) < prob
        salt = torch.rand(B, 1, H, W, device=inputs.device) < 0.5

        mask = mask.repeat(1, C, 1, 1)
        salt = salt.repeat(1, C, 1, 1)
        out[mask & salt] = 1.0
        out[mask & (~salt)] = 0.0

        inputs[t] = out

    return inputs

def add_speckle_noise(inputs: torch.Tensor, sigma: float):
    '''
    Multiplicative gaussian noise for vision sensing inputs

    Parameters
    ----------
    inputs : torch.Tensor
        input (T, B, C, H, W), float in [0, 1]
    sigma : float
        noise standard deviation
    clip : bool, optional
        clip the tensor to [0, 1], by default True
    '''
    for t in range(inputs.shape[0]):
        noise = torch.randn_like(inputs[t]) * sigma
        out = inputs[t] + inputs[t] * noise
        inputs[t] = out

    return inputs


if __name__ == '__main__':
    batch = torch.randn(2, 1, 1, 28, 28)
    print(batch)
    
    print('=' * 6 + 'gaussian noise' + '=' * 6)
    batch_aug = add_gaussian_noise(batch.clone(), sigma=0.1)
    print(batch_aug)
    
    print('=' * 6 + 'impulse noise' + '=' * 6)
    batch_aug = add_impulse_noise(batch.clone(), prob=0.3)
    print(batch_aug)

    print('=' * 6 + 'speckle noise' + '=' * 6)
    batch_aug = add_speckle_noise(batch.clone(), sigma=0.1)
    print(batch_aug)