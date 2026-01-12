import os
import datetime
import matplotlib.pyplot as plt
from rich.progress import *

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur

def tensor2numpy(x):
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
