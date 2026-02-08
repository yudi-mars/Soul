import os
import datetime
import matplotlib.pyplot as plt
import torch
from rich.progress import *
from torch import nn

def set_v_threshold(neuron, value: float):
    """
    设置神经元激发阈值

    Args:
        neuron: 神经元对象
        value: 阈值大小

    Returns:
        None
    """
    vt = getattr(neuron, "v_threshold", None)
    if isinstance(vt, torch.nn.Parameter) or torch.is_tensor(vt):
        with torch.no_grad():
            vt.fill_(float(value))
    else:
        neuron.v_threshold = float(value)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def sew_function(identity: torch.Tensor, out: torch.Tensor, cnf:str):
    if cnf == 'ADD':
        return identity + out
    elif cnf == 'AND':
        return identity * out
    elif cnf == 'IAND':
        return identity * (1. - out)
    else:
        raise NotImplementedError(cnf)

def multi_time_forward(x_seq, stateless_module):
    """
    多时间步正向推理

    Args:
        x_seq:  输入序列
        stateless_module:  推理算子

    Returns:
        推理结果
    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]  # [T, B]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)

    y_shape.extend(y.shape[1:])  # [T, B] + [...] -> [T, B, ...]
    return y.view(y_shape)

def ensure_dir(path):
    """
    路径检查/创建

    Args:
        path: 路径

    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    """
    获取本地时间

    Returns:
        本地时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def tensor2numpy(x):
    """
    tensor转numpy

    Args:
        x: 输入

    Returns:
        numpy数据
    """
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


class SpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.speed
        if speed is None:
            return Text(" ? it/s", style="progress.data.speed")
        return Text(f"{speed:5.2f} it/s", style="progress.data.speed")

def progress_bar(iterable, desc="Progress", total=None):
    """
    Wrap any iterable with a Rich progress bar
    showing elapsed time and remaining ETA.
    """
    # if no total, try to get __len__ from iterable 
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None  

    columns = [
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TextColumn("{task.completed}/{task.total}" if total else "{task.completed}"),
        TextColumn("["),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        TextColumn("•"),
        SpeedColumn(),
        TextColumn("]"),
    ]

    with Progress(*columns) as progress:
        task = progress.add_task(desc, total=total)
        for item in iterable:
            yield item
            progress.update(task, advance=1)


def set_figure_background(ax, xlabel=None, ylabel=None, title=None):
    '''
    Give some basic settings for figures
    '''
    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=14)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=12)
    
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)

    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(10)       
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    set_figure_background(axs, xlabel='X-axis', ylabel='Y-axis', title='Sample Plot')
    fig.tight_layout()
    # plt.savefig('./assets/test_plot.pdf', dpi=300, bbox_inches="tight")
    plt.show()


    # total_steps = 50
    # import time
    # for _ in progress_bar(range(total_steps), desc="Train: ", total=total_steps):
    #     time.sleep(0.05)
