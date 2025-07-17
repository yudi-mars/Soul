"""
Filename: STBIF.py
Author: Changze Lv <czlv24@m.fudan.edu.cn>
Date Created: 2025-05-15
Description:
    implementation for LIF variants for Spiking Neural Networks.

References:
    - Kang You et al., "VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network", CVPR'2025.
    https://github.com/Intelligent-Computing-Research-Group/ViStream
"""
import torch
from soul.neuron.LIF import BaseNode

class STBIF(BaseNode):
    """
    改进版 Integrate-and-Fire 神经元，兼容 BaseNode 训练 / 推理接口。
    支持量化门限和共享显存缓存。
    """

    def __init__(self, config):
        """
        config 应包含以下字段：
            - q_threshold: float or torch.Tensor, 量化门限
            - level: int, 量化等级
            - sym: bool, 是否对称量化（目前未使用）
            - store_v_seq: bool, 是否记录 v 序列（可选）
        """
        surrogate_function = config['surrogate_function']  # 仅作兼容，实际上 IF 无需 surrogate

        super().__init__(
            v_threshold=1.0,              # 实际门限由 q_threshold 决定，不直接用
            v_reset=0.0,                  # 充当硬重置行为，但已在内部控制
            surrogate_function=surrogate_function,
            detach_reset=True,
            step_mode='m',
            backend='torch',
            store_v_seq=config.get('store_v_seq', False)
        )

        self.q_threshold = torch.as_tensor(config['q_threshold'], dtype=torch.float32)
        self.level = config['level']
        self.sym = config.get('sym', False)
        self.pos_max = self.level - 1
        self.neg_min = 0
        self.eps = 0.0

        self.register_memory('q', None)          # 电荷积分
        self.register_memory('cur_output', None) # 输出脉冲

    def neuronal_charge(self, x: torch.Tensor):
        """懒初始化，并执行电荷积分（in-place）"""
        x_norm = x / self.q_threshold

        if self.q is None or self.q.shape != x.shape:
            self.q = torch.zeros_like(x, dtype=torch.float32) + 0.5
            self.cur_output = torch.zeros_like(x, dtype=x.dtype)

        with torch.no_grad():
            self.q.add_(x_norm)

    def neuronal_fire(self):
        """生成脉冲，并更新 cur_output"""
        with torch.no_grad():
            spike_pos = self.q >= 1
            self.cur_output.zero_()
            self.cur_output[spike_pos] = 1
        return self.cur_output * self.q_threshold

    def neuronal_reset(self, spike):
        """电荷回落(in-place), 无需 surrogate 重参数化"""
        with torch.no_grad():
            self.q[self.cur_output.bool()] -= 1
            
