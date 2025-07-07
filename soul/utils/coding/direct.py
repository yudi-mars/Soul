import torch

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
