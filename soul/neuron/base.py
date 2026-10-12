"""
Filename:
    base.py

Author:
    Changze Lv <czlv24@m.fudan.edu.cn>

Date Created:
    2025-04-20

Description:
    implementation of the basic fundation of spiking neurons:
        - StepModule:    single-step ('s') / multi-step ('m') inference modes;
        - MemoryModule:  stateful module whose states (memories) are restored by ``reset()``;
        - BaseNode:      charge -> fire -> reset template shared by most spiking neurons;
        - AdaptBaseNode: BaseNode with an adaptation current ``w``.

References:
    - Wei Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", Science Advances'2023.
    https://github.com/fangwei123456/spikingjelly
"""

import copy
from abc import abstractmethod
from collections.abc import Mapping

import torch
import torch.nn as nn

from soul.utils.surrogate import surrogate_map


def surrogate_from_config(config: Mapping):
    """
    从配置中获取梯度代理函数，优先使用已实例化的 ``surrogate_function``，否则按 ``surrogate`` 名称查表

    Args:
        config: 配置字典

    Returns:
        梯度代理函数
    """
    surrogate_function = config.get('surrogate_function')
    if surrogate_function is None:
        surrogate_function = surrogate_map[config['surrogate']]
    return surrogate_function


class StepModule:
    """时间步模块"""
    def supported_step_mode(self):
        """
        支持推理模式

        Returns:
            支持模式元组("s","m" or both)
        """
        return ("s", "m")

    @property
    def step_mode(self):
        """
        当前推理模式

        Returns:
            当前推理模式
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        设置推理模式

        Args:
            value: 推理模式

        Returns:
            None
        """
        if value not in self.supported_step_mode():
            raise ValueError(
                f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!'
            )
        self._step_mode = value


class MemoryModule(nn.Module, StepModule):
    """信息存储模块"""
    def __init__(self):
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = "torch"
        self.step_mode = self.supported_step_mode()[0]

    @property
    def supported_backends(self):
        """
        支持后端

        Returns:
            后端类型
        """
        return ("torch",)

    @property
    def backend(self):
        """
        获取后端类型

        Returns:
            后端类型
        """
        return self._backend

    @backend.setter
    def backend(self, value: str):
        """
        设置后端类型

        Args:
            value: 后端类型

        Returns:
            None
        """
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value} is not a supported backend of {self._get_name()}!"
            )
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        """
        多时间步推理

        Args:
            x_seq: 输入序列
            *args:  参数列表
            **kwargs: 参数字典

        Returns:
            输出序列
        """
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))
        return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        """
        正向推理

        Args:
            *args:  参数列表
            **kwargs: 参数字典

        Returns:
            输出序列
        """
        if torch.onnx.is_in_onnx_export():
            # 神经元整体导出为自定义算子 soul::<类名>，neusim 等后端按类名识别神经元，因此类名不能随意修改
            class MemoryModuleFunction(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    """前向传播"""
                    return x

                @staticmethod
                def backward(ctx, grad_output):
                    """反向传播"""
                    return grad_output

                @staticmethod
                def symbolic(g, x):
                    """ONNX 符号化"""
                    name = type(self).__name__
                    output = g.op(f"soul::{name}", x, spike_shape_i=x.type().sizes())
                    output.setType(x.type())
                    return output

            return MemoryModuleFunction.apply(*args, **kwargs)

        if self.step_mode == "s":
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == "m":
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f"step_mode={self.step_mode}, backend={self.backend}"

    def register_memory(self, name: str, value):
        """
        注册存储属性

        Args:
            name: 属性名称
            value: 属性初始值

        Returns:
            None
        """
        assert not hasattr(self, name), f"{name} has been set as a member variable!"
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        重置存储属性

        Returns:
            None
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        """
        设置存储属性重置值

        Args:
            name: 属性名称
            value: 重置值

        Returns:
            None
        """
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if "_memories" in self.__dict__:
            memories = self.__dict__["_memories"]
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get("_memories")
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories
        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]
        return sorted(keys)

    def memories(self):
        """
        获取存储属性值

        Returns:
            存储属性值迭代器
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        获取存储属性名称及值

        Returns:
            （存储属性,值）迭代器
        """
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        分离存储属性张量

        Returns:
            None
        """
        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        # do not apply on default values
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


class BaseNode(MemoryModule):
    """
    脉冲神经元基类，每个时间步依次执行：
        1. neuronal_charge(x)：用输入更新膜电位 v（由子类实现）；
        2. neuronal_fire()：发放脉冲 surrogate_function(v - v_threshold)；
        3. neuronal_reset(spike)：v_reset 不为 None 时硬重置到 v_reset，否则软重置（减去 v_threshold）。

    所有神经元都由全局配置字典构造，公共配置项（括号内为缺省值）：
        v_threshold (1.0), v_reset (0.0, None 表示软重置), surrogate_function（或 surrogate 名称）,
        detach_reset (False), store_v_seq (False), step_mode (default_step_mode)
    """
    default_step_mode = 'm'

    def __init__(self, config: Mapping):
        super().__init__()
        v_reset = config.get('v_reset', 0.)
        self.v_threshold = float(config.get('v_threshold', 1.))
        self.v_reset = None if v_reset is None else float(v_reset)
        self.detach_reset = bool(config.get('detach_reset', False))
        self.surrogate_function = surrogate_from_config(config)
        self.step_mode = config.get('step_mode', self.default_step_mode)
        self.store_v_seq = bool(config.get('store_v_seq', False))
        self.register_memory('v', 0. if self.v_reset is None else self.v_reset)

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value and not hasattr(self, 'v_seq'):
            self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike: torch.Tensor):
        if self.detach_reset:
            spike = spike.detach()

        if self.v_reset is None:
            self.v = self.jit_soft_reset(self.v, spike, self.v_threshold)
        else:
            self.v = self.jit_hard_reset(self.v, spike, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor, *args):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            # extra inputs (e.g. a second synaptic pathway) are sequences aligned with x_seq
            y = self.single_step_forward(x_seq[t], *[a[t] for a in args])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            self.v = torch.full_like(x.data, self.v)


class AdaptBaseNode(BaseNode):
    """
    带适应电流 w 的脉冲神经元基类：充电后按 w <- w + (a * (v - v_rest) - w) / tau_w 更新 w，
    每次发放后 w 增加 b。

    额外配置项：v_rest, w_rest（w 的初值）, tau_w, a（亚阈值耦合强度）, b（发放后的跳变幅度）
    """
    def __init__(self, config: Mapping):
        super().__init__(config)
        self.v_rest = float(config['v_rest'])
        self.w_rest = float(config['w_rest'])
        self.tau_w = float(config['tau_w'])
        self.a = float(config['a'])
        self.b = float(config['b'])
        self.register_memory('w', self.w_rest)

    @staticmethod
    @torch.jit.script
    def jit_neuronal_adaptation(w: torch.Tensor, tau_w: float, a: float, v_rest: float, v: torch.Tensor):
        return w + 1. / tau_w * (a * (v - v_rest) - w)

    def neuronal_adaptation(self):
        self.w = self.jit_neuronal_adaptation(self.w, self.tau_w, self.a, self.v_rest, self.v)

    def neuronal_reset(self, spike: torch.Tensor):
        super().neuronal_reset(spike)
        self.w = self.w + self.b * spike

    def extra_repr(self):
        return super().extra_repr() + f', v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.w_float_to_tensor(x)
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            self.w = torch.full_like(x.data, self.w)
