"""
Filename:
    flops.py

Author:
    Soul Framework

Date Created:
    2025-01-04

Description:
    Implementation of counting FLOPs (Floating Point Operations) for neural networks
    using the thop library.

References:
    - thop: PyTorch-OpCounter
    https://github.com/Lyken17/pytorch-OpCounter
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Callable, Union


def count_flops(
    model: nn.Module,
    input_size: Union[Tuple[int, ...], torch.Tensor],
    custom_ops: Optional[Dict[type, Callable]] = None,
) -> float:
    """
    Count the FLOPs of a PyTorch model.

    Args:
        model: The PyTorch model to profile.
        input_size: Either a tuple representing input shape (e.g., (1, 3, 224, 224))
                    or a torch.Tensor as the actual input.
        custom_ops: Optional dictionary mapping layer types to custom counting functions.

    Returns:
        Total number of floating point operations (FLOPs).

    Example:
        >>> model = torchvision.models.resnet18()
        >>> flops = count_flops(model, (1, 3, 224, 224))
        >>> print(f"FLOPs: {flops / 1e9:.2f} G")
    """
    try:
        from thop import profile
    except ImportError:
        raise ImportError(
            "thop is required for FLOPs profiling. "
            "Install it via: pip install thop"
        )

    # Prepare input tensor
    if isinstance(input_size, torch.Tensor):
        dummy_input = input_size
    else:
        dummy_input = torch.randn(*input_size)

    # Move input to same device as model
    device = next(model.parameters(), torch.tensor(0)).device
    dummy_input = dummy_input.to(device)

    # Set model to eval mode
    model.eval()

    # Profile
    if custom_ops is not None:
        flops, _ = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)
    else:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    return 2 * flops
