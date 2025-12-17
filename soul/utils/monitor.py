from typing import Optional, Union

from torch import nn


class BaseMonitor:
    def __init__(
        self, net: nn.Module, instance: Optional[Union[type, tuple[type, ...]]] = None
    ):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.input_record = None
        self.output_record = None
        self.name_records_index = {}
        self.records_name = []
        self._enable = True

        if instance is None:
            instance = type(net)
        for name, m in net.named_modules():
            if isinstance(m, instance):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(self.create_hook(name)))
            elif isinstance(m, type(net)):
                self.hooks.append(m.register_forward_hook(self.create_whole_net_hook()))

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def create_hook(self, name):
        def hook(m, x, y):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records_name.append(name)
                self.records.append(y)

        return hook

    def create_whole_net_hook(self):
        def hook(m, x, y):
            if self.is_enable():
                self.input_record = x[0]
                self.output_record = y

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()
