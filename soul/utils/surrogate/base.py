import torch
import torch.nn as nn

class SurrogateFunctionBase(nn.Module):
    """梯度代理函数基类"""
    def __init__(self, alpha, spiking=True):
        super().__init__()
        self.spiking = spiking
        self.alpha = alpha

    def set_spiking_mode(self, spiking: bool):
        """
        设置是否用于脉冲网络

        Args:
            spiking:  是否用于脉冲网络

        Returns:
            None
        """
        self.spiking = spiking

    def extra_repr(self):
        return f'alpha={self.alpha}, spiking={self.spiking}'

    @staticmethod
    def spiking_function(x, alpha):
        """
        代理函数的脉冲网络实现

        Args:
            x: 输入
            alpha: alpha值

        Returns:
            输出
        """
        raise NotImplementedError

    @staticmethod
    def primitive_function(x, alpha):
        """
        代理函数的原始实现

        Args:
            x: 输入
            alpha: alpha值

        Returns:
            输出
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if self.spiking:
            return self.spiking_function(x, self.alpha)
        else:
            return self.primitive_function(x, self.alpha)
        
@torch.jit.script
def heaviside(x: torch.Tensor):
    return (x >= 0).to(x)

class HeavisideFunction(torch.nn.Module): 
    def forward(self, x): 
        return (x > 0).float()
