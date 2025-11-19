import torch
import random

def add_gaussian_noise(x: torch.tensor, sigma: float):
    '''
    Additive gaussian noise for vision sensing inputs

    Parameters
    ----------
    x : torch.tensor
        input (T, B, C, H, W), float in [0, 1]
    sigma : float
        noise standard deviation
    '''
    for t in range(x.shape[0]):
        noise = torch.randn_like(x[t]) * sigma
        x[t] += noise

    return x

def add_impulse_noise(x: torch.Tensor, prob: float):
    '''
    Impulse noise for vision sensing inputs (a.k.a., Salt & Pepper)

    Parameters
    ----------
    inputs : torch.Tensor
        input (T, B, C, H, W), float in [0, 1]
    prob : float
        replacing probability for each pixel to be "salt & pepper"
    '''
    for t in range(x.shape[0]):
        B, C, H, W = x[t].shape

        out = x[t].clone() # (B, C, H, W)
        mask = torch.rand(B, 1, H, W, device=x.device) < prob
        salt = torch.rand(B, 1, H, W, device=x.device) < 0.5

        mask = mask.repeat(1, C, 1, 1)
        salt = salt.repeat(1, C, 1, 1)
        out[mask & salt] = 1.0
        out[mask & (~salt)] = 0.0

        x[t] = out

    return x

def add_dropouts_noise(
        x: torch.Tensor,
        drop_ratio: float = 0.1,
        block_size_ratio: tuple = (0.1, 0.1),
        fill_value: float = 0.0):
    '''
    Regional or block loss noise in simulated images (in the form of occlusion or "packet loss")

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, C, H, W), float in [0, 1]
    drop_ratio : float, optional
        The area ratio of occluded or missing blocks in the entire image, by default 0.1
    block_size_ratio : tuple, optional
        Represents the ratio of each occluded block's height and width to the image’s height and width, by default (0.1, 0.1)
    fill_value : float, optional
        The fill value for the occluded region, by default 0.0
    '''
    T, B, C, H, W = x.shape
    h_block = int(H * block_size_ratio[0])
    w_block = int(W * block_size_ratio[1])
    # Calculate the total area expected to be occluded
    total_area = H * W
    block_area = h_block * w_block
    # Estimate the number of blocks to be occluded
    num_blocks = int((drop_ratio * total_area) / (block_area + 1e-8))
    num_blocks = max(1, num_blocks)

    out = x.clone()
    for b in range(B):
        for _ in range(num_blocks):
            i0 = random.randint(0, H - h_block)
            j0 = random.randint(0, W - w_block)

            out[:, b, :, i0:i0 + h_block, j0:j0 + w_block] = fill_value

    return out

def add_speckle_noise(inputs: torch.Tensor, sigma: float):
    '''
    Multiplicative gaussian noise for vision sensing inputs

    Parameters
    ----------
    inputs : torch.Tensor
        input (T, B, C, H, W), float in [0, 1]
    sigma : float
        noise standard deviation
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

    print('=' * 6 + 'dropouts noise' + '=' * 6)
    batch_aug = add_dropouts_noise(batch.clone(), drop_ratio=0.9)
    print(batch_aug)

    # print('=' * 6 + 'speckle noise' + '=' * 6)
    # batch_aug = add_speckle_noise(batch.clone(), sigma=0.1)
    # print(batch_aug)
