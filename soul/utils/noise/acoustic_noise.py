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

def add_impulse_noise(x: torch.Tensor, prob: float, amplitude: float):
    '''
    模拟突发脉冲（点击／爆音）干扰，在谱图上随机将一些帧或通道大幅扰动。

    Parameters
    ----------
    x : torch.Tensor
        input (T, B, W, C)
    prob : float
        每个元素被扰动的概率
    amplitude : float
        扰动振幅（可正负或加减幅）
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
        time_mask_ratio: float = 0.0,
        freq_mask_ratio: float = 0.0,
        num_time_masks: int = 1,
        num_freq_masks: int = 1,
        fill_value: float = 0.0):
    '''
    对输入 mel-频谱图做时间轴遮蔽和/或频率轴遮蔽。

    Parameters
    ----------
    x : torch.Tensor
        tensor shape (B, W, C)
    time_mask_ratio : int, optional
        最大时间(单位：帧数)遮蔽比例, by default 0
    freq_mask_param : int, optional
        最大频率(单位：通道数)遮蔽比例, by default 0
    num_time_masks : int, optional
        每个样本做多少次时间遮蔽, by default 1
    num_freq_masks : int, optional
        每个样本做多少次频率遮蔽, by default 1
    fill_value : float, optional
        遮蔽填充值, by default 0.0
    '''
    T, B, W, C = x.shape
    for t in range(T):
        out = x[t].clone()
        
        # time window dropout
        if time_mask_ratio > 0:
            max_w = int(W * time_mask_ratio)
            for _ in range(num_time_masks):
                w = random.randint(0, max_w)
                w0 = random.randint(0, max(0, W - w))
                out[:, w0:w0 + w, :] = fill_value

        # frequency channel dropout
        if freq_mask_ratio > 0:
            max_f = int(C * freq_mask_ratio)
            for _ in range(num_freq_masks):
                f = random.randint(0, max_f)
                f0 = random.randint(0, max(0, C - f))
                out[:, :, f0:f0 + f] = fill_value

        x[t] = out

    return x


if __name__ == '__main__':
    batch = torch.randn(2, 1, 8, 10)
    print(batch)
    
    print('=' * 6 + 'dropout noise' + '=' * 6)
    batch_aug = add_dropouts_noise(
        batch.clone(),
        time_mask_ratio=0.10,
        freq_mask_ratio=0.20,
        num_time_masks=2,
        num_freq_masks=2,
        fill_value=0.0)
    print(batch_aug)

    print('=' * 6 + 'impulse noise' + '=' * 6)
    batch_aug = add_impulse_noise(batch.clone(), prob=0.5, amplitude=2.)
    print(batch_aug)

    print('=' * 6 + 'gaussian noise' + '=' * 6)
    batch_aug = add_gaussian_noise(batch.clone(), sigma=0.1)
    print(batch_aug)