import math
import time

import torch
import torch.distributed as dist
import torch.optim as optim
from torch import nn, Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import re
import yaml
import argparse

from soul.model.vision import spikingvgg
from soul.neuron.functional import reset_net
from soul.utils import *
from soul.model import *
from soul.neuron import neuron_map, functional


class Timer:
    def __init__(self, timer_name, logger):
        self.timer_name = timer_name
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.logger.debug('{} spent: {}.'.format(
            self.timer_name, str(datetime.timedelta(seconds=int(self.interval)))))

def parse_args():
    parser = argparse.ArgumentParser(description='arguments for soul')
    # Basic Settings
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=4,
        help="number of workers"
    )
    parser.add_argument(
        "--seed",
        "-seed",
        type=int,
        default=2025,
        help="random seed"
    )
    parser.add_argument(
        "--log_dir",
        "-log_dir",
        type=str,
        default='./logs/',
        help="save path for logs"
    )
    parser.add_argument(
        "--data_dir",
        "-data_dir",
        type=str,
        default='D:\\material\\datasets',
        help="path for input datasets"
    )
    parser.add_argument(
        "--model_dir",
        "-model_dir",
        type=str,
        default='./saved_models/',
        help="path for saved models"
    )
    parser.add_argument(
        "--epochs",
        "-epochs",
        type=int,
        default=150,
        help="number of epochs for training"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=128,
        help="batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=0.0,
        help="weight decay value"
    )
    parser.add_argument(
        "--momentum",
        "-momentum",
        type=float,
        default=0.9,
        help="inertia coefficient for optimizer"
    )
    parser.add_argument(
        "--optimizer",
        "-optimizer",
        type=str,
        default='adam',
        help="optimizer name, [optional] adam, sgd, adamw, rmsprop"
    )
    parser.add_argument(
        "--scheduler", type=str, nargs='+', default=[],
        help='''--scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
            or --scheduler Step [minestones]...''')

    # Specific Settings
    parser.add_argument(
        "--dataset_name",
        "-dataset",
        type=str,
        default='cifar10',
        help="datset name"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default='SpikingVGG9',
        help="model name"
    )
    parser.add_argument(
        "--coding_schema",
        '-coding',
        type=str,
        default='direct',
        help='encoding schema for static raw input'
    )
    parser.add_argument(
        "--neuron_type",
        "-n",
        type=str,
        default='LIF',
        help="LIF neuron type"
    )
    parser.add_argument(
        "--time_step",
        "-T",
        type=int,
        default=4,
        help="number of time steps"
    )

    # unstructured pruning
    parser.add_argument('--TET', action='store_true', help='Use TET training')
    parser.add_argument('--TET_phi', type=float, default=1.0)
    parser.add_argument('--TET_lambda', type=float, default=0.0)

    parser.add_argument('--save_latest', action='store_true')

    parser.add_argument(
        "--mode",
        "-mode",
        type=str,
        default='search',
        help="" #TODO
    )

    parser.add_argument('--not_prune_weight', action='store_true', help='not use weight prune mask')
    parser.add_argument('--not_prune_neuron', action='store_true', help='not use neuron prune mask')
    parser.add_argument('--finetune', action='store_true', help='use finetune')

    parser.add_argument('--prune_optimizer', type=str, help='adam or sgd')
    parser.add_argument('--prune_lr', type=float, help='initial learning rate of pruning')
    parser.add_argument('--prune_weight_decay', default=0, type=float)

    parser.add_argument('--penalty_lmbda', type=float, default=1e-11)

    parser.add_argument('--accumulate_step', type=int, default=1)

    parser.add_argument(
        '--temp_scheduler', type=float, nargs='+', default=[5, 1000],
        help='''--temp_scheduler <init temp> <final temp>
                    or --temp_scheduler <init temp> <final temp> <T0> <Tmax>
                    or --temp_scheduler <init temp of weight> <init temp of neuron> 
                    <final temp of weight> <final temp of neuron> <T0> <Tmax>''')

    parser.add_argument('--finetune_lr', default=1e-4, type=float, help='finetune learning rate')
    parser.add_argument('--epoch_finetune', default=200, type=int,
                        help='when to fine tune, -1 means will not fine tune')
    parser.add_argument(
        "--finetune_lr_scheduler", type=str, nargs='+', default=[],
        help='''--scheduler Cosine [<T0> <Tt> <Tmax(period of cosine)>]
                or --scheduler Step [minestones]...''')

    parser.add_argument(
        '--neuron_prune_layers', type=int, nargs='+', default=[1,2,3,4,5],
        help='''--neuron_prune_layers indicate the neuron layers need neuron pruning''')

    parser.add_argument(
        '--weight_prune_layers', type=int, nargs='+', default=[0,1,2,3,4,5,6,7],
        help='''--neuron_prune_layers indicate the conv layers need weight pruning''')

    parser.add_argument(
        '--mask_init_factor', type=float, nargs='+', default=[0, 0, 0, 0],
        help='--mask-init-factor <weights mean> <neurons mean> <weights std> <neurons std>')

    parser.add_argument('--criterion', type=str, default='MSE', help='MSE or CE')

    args = parser.parse_args()
    return args


def init_config():
    current_path = os.path.dirname(os.path.realpath(__file__))

    # load default basic yaml
    overall_init_file = os.path.join(current_path, "../../config/basic.yaml")
    config = yaml.safe_load(open(overall_init_file, 'r', encoding="utf-8"))

    # update args for user-specific settings from console
    args = parse_args()
    config.update(vars(args))

    # double-check application specific config
    dataset_name = config['dataset_name'].lower()
    if dataset_name in ['ucihar', 'hhar', 'motionsense', 'shoaib']:
        config['application'] = 'motion'
    elif dataset_name in ['cifar10', 'cifar100', 'imagenet', 'dvsgesture', 'cifar10dvs']:
        config['application'] = 'vision'
    elif dataset_name in ['gsc', 'urbansound', 'gtzan', 'ssc', 'shd']:
        config['application'] = 'acoustic'
    elif dataset_name in ['uthar', 'widar', 'fihumanid', 'fihar']:
        config['application'] = 'wireless'
    else:
        raise ValueError(f'Unsupport sensing modality: {config["dataset_name"]}')
    app_dir = config['application']

    # load neuron specific yaml
    target_config_file = os.path.join(current_path, f"../../config/neuron/{config['neuron_type'].lower()}.yaml")
    neuron_default_config = yaml.safe_load(open(target_config_file, 'r', encoding="utf-8"))
    config.update(neuron_default_config)

    # load model specific yaml
    match = re.match(r'^([a-zA-Z]+)', config['model'])
    if match:
        model_cofig_name = match.group(1)
    else:
        raise NotImplementedError(f'No yaml config for model: {config["model"]}')
    target_config_file = os.path.join(current_path, f"../../config/model/{app_dir}/{model_cofig_name.lower()}.yaml")
    model_default_config = yaml.safe_load(open(target_config_file, 'r', encoding="utf-8"))
    config.update(model_default_config)

    return config


class GlobalTimer:
    def __init__(self, timer_name, container):
        self.timer_name = timer_name
        self.container = container

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start  # seconds
        self.container[0] += self.interval

def is_distributed():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

# def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, )):
#     r"""
#     Computes the accuracy over the k top predictions for the specified values of k
#     """
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target[None])
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].flatten().sum(dtype=torch.float32)
#             res.append(correct_k * (100.0 / batch_size))
#         return res

class Record:
    r'''
    Synchronous record
    '''
    def __init__(self, test: bool = False) -> None:
        self.value = torch.tensor([0], dtype=torch.float64, device='cuda')
        self.count = torch.tensor([0], dtype=torch.int64, device='cuda')
        self.global_value = 0.0
        self.global_count = 0
        self.test = test

    def sync(self) -> None:
        r'''
        reduce value and count, and update global ones
        '''
        if is_distributed() and not self.test:
            torch.distributed.all_reduce(self.value, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(self.count, torch.distributed.ReduceOp.SUM)
        self.global_value += self.value.item()
        self.global_count += self.count.item()
        self.value[0] = 0.0
        self.count[0] = 0

    def update(self, value, count=1) -> None:
        r'''
        update local value and count
        '''
        self.value[0] += value * count
        self.count[0] += count

    def reset(self) -> None:
        self.value[0] = 0.0
        self.count[0] = 0
        self.global_value = 0.0
        self.global_count = 0

    @property
    def ave(self):
        if self.global_count == 0:
            return math.nan
        return self.global_value / self.global_count

class RecordDict:
    def __init__(self) -> None:
        self.__inner_dict = dict()

    def __init__(self, dic: dict, test: bool = False) -> None:
        self.__inner_dict = dict()
        self.test = test
        for key in dic.keys():
            self.__inner_dict[key] = Record(test)

    def __getitem__(self, key) -> Record:
        return self.__inner_dict[key]

    def __setitem__(self, key, value) -> None:
        assert (isinstance(value, Record))
        self.__inner_dict[key] = value

    def __str__(self) -> str:
        s = []
        for key, value in self.__inner_dict.items():
            s.append('{key}:{value}'.format(key=key, value=value.ave))
        return ', '.join(s)

    def sync(self):
        for value in self.__inner_dict.values():
            value.sync()

    def reset(self):
        for value in self.__inner_dict.values():
            value.reset()

    def add_record(self, key):
        self.__inner_dict[key] = Record(self.test)

class CriterionWarpper(nn.Module):
    def __init__(self, criterion, TET=False, TET_phi=1.0, TET_lambda=0.0) -> None:
        super().__init__()
        self.criterion = criterion
        self.TET = TET
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        self.mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        if self.TET:
            loss = 0
            for t in range(output.shape[0]):
                loss = loss + (1. - self.TET_lambda) * self.criterion(output[t], target)
            loss = loss / output.shape[0]
            if self.TET_lambda != 0:
                loss = loss + self.TET_lambda * self.mse(
                    output,
                    torch.zeros_like(output).fill_(self.TET_phi))
            return loss
        else:
            return self.criterion(output, target)


class UnstructuredPruningManager:
    def __init__(self, model,cfg, sample):
        self.model = model
        self.config = cfg
        self._mask_init_factor = cfg['mask_init_factor']
        self.neuron_type = cfg['neuron_type']
        self.weight_prune_layers = cfg['weight_prune_layers']
        self.neuron_prune_layers = cfg['neuron_prune_layers']
        self.weight_masks = []
        self.neuron_masks = []
        self.weight_hooks = []
        self.neuron_hooks = []
        self.weight_prune_mapper = {}
        self.neuron_prune_mapper = {}
        self.pruning = False
        self.weight_lmbdas = []
        self.weight_temps = []
        self.neuron_lmbdas = []
        self.neuron_temps = []
        self.sample = sample

        self.init_weight()
        self.init_masks()
        reset_net(self.model)

    def set_pruning(self,state:bool):
        self.pruning = state

    def init_weight(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_masks(self):
        conv_count,if_count = 0,0

        sample = self.sample.transpose(0, 1)

        _ = self.model(sample)

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                if conv_count in self.weight_prune_layers:
                    self.weight_lmbdas.append(0.0)
                    self.weight_temps.append(1.0)

                    mask_value = torch.nn.parameter.Parameter(
                        torch.normal(self._mask_init_factor[0], self._mask_init_factor[2], size=module.weight.shape,
                                     device='cuda'))
                    self.weight_masks.append(mask_value)

                    hook = self._register_weight_hook(module,len(self.weight_masks) - 1)
                    self.weight_hooks.append(hook)

                    self.weight_prune_mapper[conv_count] = len(self.weight_hooks) - 1

                conv_count += 1
            if isinstance(module, neuron_map[self.neuron_type.lower()]):
                if if_count in self.neuron_prune_layers:
                    self.neuron_lmbdas.append(0.0)
                    self.neuron_temps.append(1.0)

                    mask_value = torch.nn.parameter.Parameter(
                        torch.normal(self._mask_init_factor[1], self._mask_init_factor[3], size=module.v.unsqueeze(0).shape,
                                     device='cuda'))
                    self.neuron_masks.append(mask_value)

                    hook = self._register_neuron_hook(module, len(self.neuron_masks) - 1)
                    self.neuron_hooks.append(hook)

                    self.neuron_prune_mapper[if_count] = len(self.neuron_hooks) - 1
                if_count += 1

    def _register_weight_hook(self, layer, mask_idx):
        def weight_function_hook(module, input, output):
            mask = self.weight_masks[mask_idx]
            temp = self.weight_temps[mask_idx]

            if self.pruning:
                weight = torch.sigmoid(temp * mask) * layer.weight
            else:
                weight = torch.where(mask > 0, layer.weight, 0)

            new_output = torch.nn.functional.conv2d(input[0], weight, bias=layer.bias, stride=layer.stride,
                                           padding=layer.padding, dilation=layer.dilation,
                                           groups=layer.groups)

            return new_output

        hook = layer.register_forward_hook(weight_function_hook)

        return hook

    def remove_hook(self):
        for hook in self.weight_hooks:
            hook.remove()
        for hook in self.neuron_hooks:
            hook.remove()

    def re_register_hook(self):
        self.weight_hooks.clear()
        self.neuron_hooks.clear()

        conv_count, if_count= 0, 0

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                if conv_count in self.weight_prune_layers:

                    hook = self._register_weight_hook(module, self.weight_prune_mapper[conv_count])
                    self.weight_hooks.append(hook)

                conv_count += 1
            if isinstance(module, neuron_map[self.neuron_type.lower()]):
                if if_count in self.neuron_prune_layers:

                    hook = self._register_neuron_hook(module, self.neuron_prune_mapper[if_count])
                    self.neuron_hooks.append(hook)

                if_count += 1

    def _register_neuron_hook(self, layer, mask_idx):
        def neuron_function_hook(module, input, output):
            mask = self.neuron_masks[mask_idx]
            temp = self.neuron_temps[mask_idx]

            if self.pruning:
                new_output = torch.sigmoid(temp * mask) * output
            else:
                new_output = torch.where(mask > 0, output, 0)

            return new_output

        hook = layer.register_forward_hook(neuron_function_hook)

        return hook

    def _weight_left(self,idx,mask):
        if self.pruning:
            return  torch.sigmoid(self.weight_temps[idx] * mask.detach()).sum().item(),mask.numel()
        else:
            return torch.where(mask > 0, 1, 0).float().sum().item(), mask.numel()

    def left_weights(self):
        conn = 0
        total = 0
        for i, m in enumerate(self.weight_masks):
            c, t = self._weight_left(i,m)
            conn, total = conn + c, total + t

        return conn, total

    def _neuron_left(self,idx,mask):
        if self.pruning:
            return  torch.sigmoid(self.neuron_temps[idx] * mask.detach()).sum().item(),mask.numel()
        else:
            return torch.where(mask > 0, 1, 0).float().sum().item(), mask.numel()

    def left_neurons(self):
        conn = 0
        total = 0
        for i, m in enumerate(self.neuron_masks):
            c, t = self._neuron_left(i,m)
            conn, total = conn + c, total + t

        return conn, total


class PenaltyTerm(nn.Module):
    def __init__(self, model: nn.Module, lmbda: float, manager: UnstructuredPruningManager,device) -> None:
        super(PenaltyTerm, self).__init__()
        self.model = model
        self.lmbda = lmbda
        self.manager = manager
        self.device = device
        self.calc_c()


    def forward(self) -> Tensor:
        loss = 0

        for i, mask in enumerate(self.manager.weight_masks,start=0):
            loss = loss + (self.manager.weight_lmbdas[i] * self.lmbda) * (torch.sigmoid(
                    mask * self.manager.weight_temps[i])).sum()
        for i, mask in enumerate(self.manager.neuron_masks, start=0):
            loss = loss + (self.manager.neuron_lmbdas[i] * self.lmbda) * (torch.sigmoid(
                mask * self.manager.neuron_temps[i])).sum()
        return loss

    def calc_c(self):
        conv_count, if_count= 0, 0
        self.manager.remove_hook()
        sample = self.manager.sample
        sample = sample.transpose(0, 1)
        x = sample.flatten(0,1).to(self.device)
        pre_layers = []

        if self.manager.config['model'] in spikingvgg.__dict__:
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    assert module.stride[0] == module.stride[1]
                    stride = module.stride[0]
                    assert module.dilation[0] == module.dilation[1] == 1
                    assert module.groups == 1


                    with torch.no_grad():
                        if conv_count not in self.manager.weight_prune_layers:
                            conv_count += 1
                            continue

                        self.manager.weight_lmbdas[self.manager.weight_prune_mapper[conv_count]] = x.shape[-2] * x.shape[-1] / (stride * stride)
                        c_prev = module.weight.shape[0] * module.weight.shape[2] * module.weight.shape[
                            3] / (stride * stride)
                        for layer in pre_layers:
                            if len(self.manager.neuron_lmbdas) <= layer:
                                break
                            self.manager.neuron_lmbdas[layer] += c_prev
                        pre_layers.clear()

                        if x.shape[-2] < module.kernel_size[0] or x.shape[-1] < module.kernel_size[1]:
                            output_size = (max(module.kernel_size[0], x.shape[-1]), max(module.kernel_size[0], x.shape[-1]))
                            x = torch.nn.functional.adaptive_avg_pool2d(x,
                                                      output_size)  # -> (TB, C, output_size[0], output_size[1])

                        y = torch.nn.functional.conv2d(x, module.weight, None, module.stride,
                                                       module.padding, module.dilation, module.groups)
                        x = torch.zeros_like(y)

                        conv_count += 1
                if isinstance(module,nn.AdaptiveAvgPool2d) or isinstance(module,nn.AvgPool2d) or isinstance(module,nn.MaxPool2d):
                    with torch.no_grad():
                        x = module(x)
                if isinstance(module, neuron_map[self.manager.config['neuron_type'].lower()]):
                    pre_layers.append(if_count)
                    if_count += 1
        self.manager.re_register_hook()

class TemperatureScheduler:
    def __init__(self, model: nn.Module, init_temp: float, final_temp: float, T0: int,
                 Tmax: int, manager:UnstructuredPruningManager) -> None:
        assert init_temp > 0
        assert init_temp <= final_temp
        assert T0 < Tmax
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.T0 = T0
        self.Tmax = Tmax
        self.current_step = 0
        self.factor = math.log(self.final_temp / self.init_temp)
        self.manager = manager

    def _temp(self):
        if self.current_step < self.T0:
            return self.init_temp
        if self.current_step > self.Tmax:
            return self.final_temp
        return self.init_temp * math.exp(self.factor * (self.current_step - self.T0) /
                                         (self.Tmax - self.T0))

    def step(self):
        self.current_step = self.current_step + 1
        temp = self._temp()
        for i in range(0, len(self.manager.weight_temps)):
            self.manager.weight_temps[i] = temp

        for i in range(0, len(self.manager.neuron_temps)):
            self.manager.neuron_temps[i] = temp
        return

    def __str__(self) -> str:
        return 'temperature: {:.3e}'.format(self._temp())


class SplitTemperatureScheduler:
    def __init__(self, model: nn.Module, init_temp_w: float, init_temp_n: float,
                 final_temp_w: float, final_temp_n: float, T0: int, Tmax: int, manager:UnstructuredPruningManager) -> None:
        assert init_temp_w > 0 and init_temp_n > 0
        assert init_temp_w <= final_temp_w and init_temp_n <= final_temp_n
        assert T0 < Tmax
        self.init_temp_w = init_temp_w
        self.final_temp_w = final_temp_w
        self.init_temp_n = init_temp_n
        self.final_temp_n = final_temp_n
        self.T0 = T0
        self.Tmax = Tmax
        self.current_step = 0
        self.factor_w = math.log(self.final_temp_w / self.init_temp_w)
        self.factor_n = math.log(self.final_temp_n / self.init_temp_n)
        self.manager = manager

    def _temp_w(self):
        if self.current_step < self.T0:
            return self.init_temp_w
        if self.current_step > self.Tmax:
            return self.final_temp_w
        return self.init_temp_w * math.exp(self.factor_w * (self.current_step - self.T0) /
                                           (self.Tmax - self.T0))

    def _temp_n(self):
        if self.current_step < self.T0:
            return self.init_temp_n
        if self.current_step > self.Tmax:
            return self.final_temp_n
        return self.init_temp_n * math.exp(self.factor_n * (self.current_step - self.T0) /
                                           (self.Tmax - self.T0))

    def step(self):
        self.current_step = self.current_step + 1
        temp_w = self._temp_w()
        temp_n = self._temp_n()
        for i in range(0,len(self.manager.weight_temps)):
            self.manager.weight_temps[i] = temp_w

        for i in range(0,len(self.manager.neuron_temps)):
            self.manager.neuron_temps[i] = temp_n
        return

    def __str__(self) -> str:
        return 'temperature: weight: {:.3e}, neuron: {:.3e}'.format(self._temp_w(), self._temp_n())


def train_one_epoch(model, criterion, penalty_term, optimizer_train, optimizer_prune,
                    data_loader_train, temp_scheduler,accumulate_step=1, prune=False, one_hot=None,manager=None):
    model.train()
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    timer_container = [0.0]

    manager.set_pruning(prune)
    model.zero_grad()
    for idx, (image, target) in enumerate(data_loader_train):
        with GlobalTimer('iter', timer_container):
            image, target = image.float().cuda(), target.cuda()

            image = image.transpose(0, 1)

            output = model(image)
            if one_hot:
                loss = criterion(output, torch.nn.functional.one_hot(target.long(), one_hot).float())
            else:
                loss = criterion(output, target)

            metric_dict['loss'].update(loss.item())
            if prune:
                loss = loss + penalty_term()

            loss = loss / accumulate_step

            loss.backward()
            if (idx + 1) % accumulate_step == 0:
                if prune:
                    optimizer_prune.step()
                optimizer_train.step()
                model.zero_grad()
                if temp_scheduler is not None:
                    temp_scheduler.step()

            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_s = acc1.item()
            acc5_s = acc5.item()

            batch_size = image.shape[1]
            metric_dict['acc@1'].update(acc1_s, batch_size)
            metric_dict['acc@5'].update(acc5_s, batch_size)

    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave

def evaluate(model, criterion, data_loader, prune, one_hot,manager):
    model.eval()
    manager.set_pruning(prune)
    metric_dict = RecordDict({'loss': None, 'acc@1': None, 'acc@5': None})
    with torch.no_grad():
        for idx, (image, target) in enumerate(data_loader):
            image = image.float().to(torch.device('cuda'), non_blocking=True)
            target = target.to(torch.device('cuda'), non_blocking=True)

            image = image.transpose(0, 1)

            output = model(image)
            if one_hot:
                loss = criterion(output, torch.nn.functional.one_hot(target.long(), one_hot).float())
            else:
                loss = criterion(output, target)
            metric_dict['loss'].update(loss.item())
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # could have been padded in distributed setup
            batch_size = image.shape[1]
            metric_dict['acc@1'].update(acc1.item(), batch_size)
            metric_dict['acc@5'].update(acc5.item(), batch_size)

    metric_dict.sync()
    return metric_dict['loss'].ave, metric_dict['acc@1'].ave, metric_dict['acc@5'].ave

if __name__ == '__main__':
    config = init_config()

    # activate distributed
    config['is_distributed'] = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if config['is_distributed']:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # gpu for current process
        device = torch.device("cuda", local_rank)
        # main process
        global_rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        global_rank = 0

    # init logger
    if global_rank == 0:
        log_path = os.path.join(
            config['log_dir'],
            config['dataset_name'].lower(),
            config['model'].lower(),
            config['neuron_type'].lower()
        )
        ensure_dir(log_path)
        logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])
        logger.info(f'Distributed Training: {config["is_distributed"]}')
    else:
        logger = None

    # report all configuration
    for k, v in sorted(config.items()):
        if global_rank == 0:
            logger.info(f'{k} = {v}')

    # reproducibility
    if global_rank == 0:
        logger.info(f'Reproducibility with random seed {config["seed"]}')
        init_seed(config["seed"])
        logger.info('=' * 50)

    # load data
    if global_rank == 0:
        logger.info('Load data...')
    train_dataset, test_dataset = load_dataset(config)

    if config['is_distributed']:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset)
        # define the batch size per gpu, usually we define the numer of process equal to the number of used gpus
        world_size = dist.get_world_size()
        config['batch_size'] //= world_size
    else:
        train_sampler = None

    # load dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False if config['is_distributed'] else True,
        sampler=train_sampler,
        num_workers=config['workers'],
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['workers'],
        pin_memory=True
    )

    # load SNN model
    if global_rank == 0:
        logger.info(f'Load SNN model: {config["model"]} featured {config["neuron_type"].upper()} neuron...')

    if global_rank == 0:
        logger.debug(f'surrogate function: {config["surrogate"]}')
    config['surrogate_function'] = surrogate_map[config['surrogate']]
    config['neuron'] = neuron_map[config['neuron_type'].lower()](config)

    model = model_map[config['application']][config['model'].lower()](config)

    param_without_masks = list(model.parameters())

    # init model optimizer
    if config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(param_without_masks, lr=config['learning_rate'], momentum=0.9,
                              weight_decay=config['weight_decay'], nesterov=True)
    elif config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(param_without_masks, lr=config['learning_rate'],
                               betas=(0.9, 0.999), weight_decay=config['weight_decay'])
    else:
        raise ValueError(config['optimizer'])

    # init mask
    if config['dataset_name'] == 'cifar10':
        one_hot =10
        sample = torch.rand(1,config['time_step'], 3, 32, 32).cuda()
    elif config['dataset_name'] == 'cifar100':
        one_hot =100
        sample = torch.rand(1,config['time_step'], 3, 32, 32).cuda()

    model.to(device)
    pruningManager = UnstructuredPruningManager(model, config, sample)
    masks = pruningManager.neuron_masks+pruningManager.weight_masks

    #init prune optimizer
    if not (config['not_prune_weight'] and config['not_prune_neuron']):
        if config['prune_optimizer'] is None:
            config['prune_optimizer'] = config['optimizer']
        if config['prune_lr'] is None:
            config['prune_lr'] = config['learning_rate']
        if config['prune_optimizer'] == 'sgd':
            optimizer_prune = torch.optim.SGD(masks, lr=config['prune_lr'], momentum=0.9,
                                              weight_decay=config['prune_weight_decay'], nesterov=True)
        elif config['prune_optimizer'] == 'adam':
            optimizer_prune = torch.optim.Adam(masks, lr=config['prune_lr'], betas=(0.9, 0.999),
                                               weight_decay=config['prune_weight_decay'])
        else:
            raise ValueError(config['prune_optimizer'])

    # criterion and penalty
    if config['criterion'] == 'MSE':
        criterion = nn.MSELoss()
    elif config['criterion'] == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(config['criterion'])
    criterion = CriterionWarpper(criterion, config['TET'], config['TET_phi'], config['TET_lambda'])

    if not (config['not_prune_weight'] and config['not_prune_neuron']):
        penalty_term = PenaltyTerm(model, config['penalty_lmbda'],pruningManager,device)

    # lr scheduler
    milestones = []
    lr_scheduler_train, lr_scheduler_prune = None, None
    lr_scheduler_T0, lr_scheduler_Tmax = 0, config['epochs']
    if not (config['not_prune_weight'] and config['not_prune_neuron']):
        if len(config['scheduler'] ) != 0:
            if config['scheduler'][0] == 'Step':
                for i in range(1, len(config['scheduler'])):
                    milestones.append(int(config['scheduler'][i]))
                lr_scheduler_train = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=milestones, gamma=0.1)
                lr_scheduler_prune = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer_prune, milestones=milestones, gamma=0.1)
            elif config['scheduler'][0] == 'Cosine':
                if len(config['scheduler']) > 1:
                    lr_scheduler_T0, lr_scheduler_Tmax, T_max = int(
                        config['scheduler'][1]), int(config['scheduler'][2]), int(
                        config['scheduler'][3])
                else:
                    T_max = lr_scheduler_Tmax - lr_scheduler_T0
                lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=T_max)
                lr_scheduler_prune = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer_prune, T_max=T_max)
            else:
                raise ValueError(config['scheduler'])

    model_without_ddp = model
    if config['is_distributed']:
        model = DDP(model, device_ids=[local_rank])
        model_without_ddp = model.module

    # threshold scheduler
    if not (config['not_prune_weight'] and config['not_prune_neuron']):
        iter_per_epoch = len(train_loader) // config['accumulate_step']
        if len(config['temp_scheduler']) == 2:
            (config['temp_scheduler']).append(0)
            (config['temp_scheduler']).append(config['epochs'])
        if len(config['temp_scheduler']) == 4:
            temp_scheduler = TemperatureScheduler(model, config['temp_scheduler'][0],
                                                  config['temp_scheduler'][1],
                                                  int(config['temp_scheduler'][2]) * iter_per_epoch,
                                                  int(config['temp_scheduler'][3]) * iter_per_epoch,
                                                  pruningManager)
        elif len(config['temp_scheduler']) == 6:
            temp_scheduler = SplitTemperatureScheduler(model, config['temp_scheduler'][0],
                                                       config['temp_scheduler'][1],
                                                       config['temp_scheduler'][2],
                                                       config['temp_scheduler'][3],
                                                       int(config['temp_scheduler'][4]) * iter_per_epoch,
                                                       int(config['temp_scheduler'][5]) * iter_per_epoch,
                                                  pruningManager)
        else:
            raise ValueError(config['temp_scheduler'])

    start_epoch = 0
    max_acc1 = 0

    logger.debug(str(model))

    if config['mode'] == 'search':
        logger.info("Search start")
        for epoch in range(start_epoch, config['epochs']):
            if config['is_distributed']:
                train_sampler.set_epoch(epoch)
            logger.info('Epoch [{}] Start, lr {:.6f}, {}'.format(epoch,
                                                                 optimizer.param_groups[0]["lr"],
                                                                 str(temp_scheduler)))

            with Timer(' Train', logger):
                logger.debug('[Training]')
                train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, penalty_term, optimizer, optimizer_prune, train_loader,
                temp_scheduler,config['accumulate_step'], True, one_hot, pruningManager)
                if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                    lr_scheduler_train.step()
                    lr_scheduler_prune.step()

            with Timer(' Test', logger):
                logger.debug('[Test with continuous mask]')
                test_loss_c, test_acc1_c, test_acc5_c = evaluate(model, criterion, test_loader, True, one_hot,pruningManager)
                logger.info(' Test (continuous mask) Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
                    test_acc1_c, test_acc5_c))

                logger.debug('[Test with binary mask]')
                test_loss_s, test_acc1_s, test_acc5_s = evaluate(model, criterion, test_loader, False, one_hot,pruningManager)
                logger.info(' Test (binary mask) Acc@1: {:.5f}, Acc@5: {:.5f}'.format(
                    test_acc1_s, test_acc5_s))

            pruningManager.set_pruning(True)
            n_l, n_t = pruningManager.left_neurons()
            w_l, w_t = pruningManager.left_weights()
            neu, wei = 100 * (n_l + 1e-10) / (n_t + 1e-10), 100 * (w_l + 1e-10) / (w_t + 1e-10)

            logger.info(' left neurons: {:.2f}%, left weights: {:.2f}%'.format(
                neu, wei))

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer_train': optimizer.state_dict(),
                'optimizer_prune': optimizer_prune.state_dict(),
                'epoch': epoch }

            if lr_scheduler_train is not None:
                checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()
                checkpoint['lr_scheduler_prune'] = lr_scheduler_prune.state_dict()

            if config['save_latest']:
                torch.save(checkpoint,
                           os.path.join(config['model_dir'],
                                        f'up_search_lastest_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pth'))

            if (epoch + 1) == config['epochs']:
                torch.save(checkpoint,
                           os.path.join(config['model_dir'],
                                        f'up_search_sparsified_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pth'))
        logger.info('Search finish.')

    if config['finetune']:
        if config['finetune_lr'] is None:
            config['finetune_lr'] = config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['finetune_lr']

        # lr scheduler
        milestones = []
        lr_scheduler_train, lr_scheduler_prune = None, None
        lr_scheduler_T0, lr_scheduler_Tmax = 0, config['epochs']
        if len(config['finetune_lr_scheduler']) != 0:
            if config['finetune_lr_scheduler'][0] == 'Step':
                for i in range(1, len(config['finetune_lr_scheduler'])):
                    milestones.append(int(config['finetune_lr_scheduler'][i]))
                lr_scheduler_train = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, milestones=milestones, gamma=0.1)
            elif config['finetune_lr_scheduler'][0] == 'Cosine':
                if len(config['finetune_lr_scheduler']) > 1:
                    lr_scheduler_T0, lr_scheduler_Tmax, T_max = int(
                        config['finetune_lr_scheduler'][1]), int(config['finetune_lr_scheduler'][2]), int(
                        config['finetune_lr_scheduler'][3])
                else:
                    T_max = lr_scheduler_Tmax - lr_scheduler_T0
                lr_scheduler_train = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, T_max=T_max)
            else:
                raise ValueError(config['finetune_lr_scheduler'])

        start_epoch = 0
        logger.info("Finetune start")

        for epoch in range(start_epoch, config['epoch_finetune']):
            save_max = False
            if config['is_distributed']:
                train_sampler.set_epoch(epoch)
            logger.info('Epoch [{}] Start, lr {:.6f}'.format(epoch,
                                                             optimizer.param_groups[0]["lr"]))

            with Timer(' Train', logger):
                logger.debug('[Training]')
                train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, None, optimizer,
                                                        None, train_loader, None,config['accumulate_step'],
                                                         False, one_hot,pruningManager)

                if lr_scheduler_train is not None and lr_scheduler_T0 <= epoch < lr_scheduler_Tmax:
                    lr_scheduler_train.step()

            with Timer(' Test', logger):
                logger.debug('[Test]')
                test_loss, test_acc1, test_acc5 = evaluate(model, criterion, test_loader, False, one_hot,pruningManager)

            logger.info(' Test Acc@1: {:.5f}, Acc@5: {:.5f}'.format(test_acc1, test_acc5))
            if max_acc1 < test_acc1:
                max_acc1 = test_acc1
                save_max = True

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer_train': optimizer.state_dict(),
                'epoch': epoch}

            if lr_scheduler_train is not None:
                checkpoint['lr_scheduler_train'] = lr_scheduler_train.state_dict()

            if config['save_latest']:
                torch.save(checkpoint,
                           os.path.join(config['model_dir'],
                                        f'up_finetune_lastest_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pth'))

            if (epoch + 1) == config['epochs']:
                torch.save(checkpoint,
                           os.path.join(config['model_dir'],
                                        f'up_finetune_sparsified_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pth'))

        logger.info('Finetune finish.')