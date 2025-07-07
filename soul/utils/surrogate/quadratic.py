import torch
from .base import SurrogateFunctionBase, heaviside

@torch.jit.script
def piecewise_quadratic_backward(grad_output: torch.Tensor, x: torch.Tensor, alpha: float):
    x_abs = x.abs()
    mask = (x_abs > (1 / alpha))
    grad_x = (grad_output * (- (alpha ** 2) * x_abs + alpha)).masked_fill_(mask, 0)
    return grad_x, None

class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return piecewise_quadratic_backward(grad_output, ctx.saved_tensors[0], ctx.alpha)

class PieceWiseQuadratic(SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)

    @staticmethod
    def spiking_function(x, alpha):
        return piecewise_quadratic.apply(x, alpha)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, alpha: float):
        mask0 = (x > (1.0 / alpha)).to(x)
        mask1 = (x.abs() <= (1.0 / alpha)).to(x)

        return mask0 + mask1 * (-(alpha ** 2) / 2 * x.square() * x.sign() + alpha * x + 0.5)

    @staticmethod
    def backward(grad_output, x, alpha):
        return piecewise_quadratic_backward(grad_output, x, alpha)[0]