"""
Wrapper for NeuSim simulator
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['SimResult', 'run']
class SimResult:
    def __init__(self, arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex, arg2: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def duration(self) -> float:
        ...
    @duration.setter
    def duration(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def num_timesteps(self) -> int:
        ...
    @num_timesteps.setter
    def num_timesteps(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def retcode(self) -> int:
        ...
    @retcode.setter
    def retcode(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_cycles(self) -> int:
        ...
    @total_cycles.setter
    def total_cycles(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_firing_cnt(self) -> int:
        ...
    @total_firing_cnt.setter
    def total_firing_cnt(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_hops(self) -> int:
        ...
    @total_hops.setter
    def total_hops(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_recv_flits(self) -> int:
        ...
    @total_recv_flits.setter
    def total_recv_flits(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_recv_spikes(self) -> int:
        ...
    @total_recv_spikes.setter
    def total_recv_spikes(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_sent_spikes(self) -> int:
        ...
    @total_sent_spikes.setter
    def total_sent_spikes(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def total_update_cnt(self) -> int:
        ...
    @total_update_cnt.setter
    def total_update_cnt(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
def run(position: typing.Annotated[numpy.typing.ArrayLike, numpy.uint32], core_conns: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], spikes: typing.Annotated[numpy.typing.ArrayLike, numpy.uint8], **kwargs) -> SimResult:
    """
    NeuSim
    """
