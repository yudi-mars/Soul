import torch
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import re
import yaml
import argparse
from soul.utils import *
from soul.model import *
from soul.neuron import neuron_map


def parse_args():
    parser = argparse.ArgumentParser(description='arguments for soul')
    # Basic Settings
    parser.add_argument(
        "--mode",
        "-mode",
        type=str,
        default='train',
        help="'train' or 'load' mean:train,prune and test or load pretrained model from specific path,then prune and test"
    )
    parser.add_argument(
        "--model_path",
        "-model_path",
        type=str,
        default='./saved_models/best_spikingvgg16_lif_cifar10_2025.pt',
        help="load pretrained model from this path if mode is 'load'"
    )
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
        default='D:\\material\\datasets\\cifar-10-batches-py',
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
        default=50,
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
        "--scheduler",
        "-scheduler",
        type=str,
        default='cosine',
        help="scheduler name, [optional] cosine, linear, warmup"
    )
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
        default='SpikingVGG16',
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
    parser.add_argument(
        "--pruning_nodes",
        "-nodes",
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        help="index of nodes will be pruned"
    )

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

    # load neuron specific yaml TODO neuron also need specify application scenario??
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

class PruningManager:
    def __init__(self,model:nn.Module,nodes_arr:list[int]):
        self.masks = []
        self.activations = []
        self.all_num = 0.0
        self.grow_num = 0.0
        self.prune_num = 0.0
        self.ifnodes = []
        self.post_layers = []
        self.pre_layers= []
        self.hooks = []
        self.model = model
        self._hooked_ifnodes_num = len(nodes_arr)
        self._nodes_arr = nodes_arr
        self._hook_function = None
        self._pre_conv, self._pre_bn, self._pre_layer = None, None, None
        self.scan_model()
        self.training = False
        self._res_type = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def check_ifnode(self, node,i):
        if i not in self._nodes_arr:
            return False
        if hasattr(node, 'store_v_seq') and node.store_v_seq:
            for val in neuron_map.values():
                if isinstance(node, val):
                        return True

        return False


    def scan_model(self):
        i = 0
        pre_ifnode_idx = -1
        _pre_nodes = []

        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                if pre_ifnode_idx != -1:
                    if pre_ifnode_idx >= len(self.post_layers):
                        self.post_layers.append(layer)
                    else:
                        self.post_layers[pre_ifnode_idx] = layer
                    pre_ifnode_idx = -1

            if isinstance(layer, nn.Conv2d):
                if pre_ifnode_idx != -1:
                    if pre_ifnode_idx >= len(self.post_layers):
                        self.post_layers.append(layer)
                    else:
                        self.post_layers[pre_ifnode_idx] = layer
                    pre_ifnode_idx = -1
                _pre_nodes.append(layer)

            if isinstance(layer, nn.BatchNorm2d):
                _pre_nodes.append(layer)

            if self.check_ifnode(layer,i):
                self.register_pruning_hook(layer, i)

                if len(self.ifnodes) < self._hooked_ifnodes_num:
                    self.ifnodes.append(layer)
                    self.pre_layers.append(_pre_nodes)
                else:
                    self.ifnodes[i] = layer
                    self.pre_layers[i] = _pre_nodes

                _pre_nodes = []
                pre_ifnode_idx = i
                i += 1

    def register_pruning_hook(self, layer, idx):
        def default_hook_function(model, input, output):
            if model.training:
                spikes = output.detach()
                membrance = model.v_seq.abs().detach()
                v_temp = (spikes + membrance).detach()
                if len(self.activations) < self._hooked_ifnodes_num:
                    self.activations.append(torch.mean(v_temp, dim=(0, 1, 3, 4)))
                else:
                    self.activations[idx] = torch.mean(v_temp, dim=(0, 1, 3, 4))

        handle = layer.register_forward_hook(default_hook_function)
        if len(self.activations) < self._hooked_ifnodes_num:
            self.hooks.append(handle)
        else:
            self.hooks[idx] = handle

    def calculate_masks(self, model, remove_threshold = 0.8, grow_threshold = 0.1):
        sorted_indices = torch.argsort(torch.cat(self.activations), descending=True)
        num_elements = int(len(sorted_indices) * remove_threshold)
        threshold_indices = sorted_indices[num_elements]
        threshold_prune = torch.cat(self.activations)[threshold_indices]

        for i in range(len(self.activations)):
            if i >= len(self.masks):
                mask=(self.activations[i] > threshold_prune).float().detach()
                self.masks.append(mask)
            else:
                self.masks[i]=(self.activations[i] > threshold_prune).float().detach()

        i = 0
        channel_gradients_grows = []
        for module in model.modules():
            if i >= len(self.masks):
                break
            if isinstance(module, nn.BatchNorm2d):
                gradients_grow = torch.where(self.masks[i] == 0, torch.abs(module.weight.grad),
                                             torch.zeros_like(module.weight.grad))
                channel_gradients_grows.append(gradients_grow)
                i += 1

        sorted_indices_grow = torch.argsort(torch.cat(channel_gradients_grows), descending=True)
        num_elements_grow = int(len(sorted_indices_grow) * grow_threshold)
        threshold_indice_grow = sorted_indices_grow[num_elements_grow]
        threshold_grow = torch.cat(channel_gradients_grows)[threshold_indice_grow]

        for i in range(len(self.masks)):
            self.masks[i][channel_gradients_grows[i] > threshold_grow] = 1

    def dummy_pruning(self):
        i = 0
        for layers in self.pre_layers:
            prune_indices = (self.masks[i] == 0).nonzero().view(-1)

            for layer in layers:
                mask_l = torch.ones_like(layer.weight.data)
                if isinstance(layer, nn.Conv2d):
                    mask_l[prune_indices, :, :, :] = 0
                    layer.weight.data.mul_(mask_l)

                if isinstance(layer, nn.BatchNorm2d):
                    mask_l[prune_indices] = 0
                    layer.weight.data.mul_(mask_l)
                    layer.bias.data.mul_(mask_l)
            i += 1

    def structured_pruning(self):
        i = 0
        for _ in self.ifnodes:
            layers = self.pre_layers[i]
            post_layer = self.post_layers[i]

            prune_indices = (self.masks[i] == 0).nonzero().view(-1)
            prune_num = len(prune_indices)
            final_channel_num = 0

            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    final_channel_num = layer.weight.data.shape[0] - prune_num
                    device = layer.weight.data.device
                    final_data = torch.ones(
                        (final_channel_num, layer.weight.data.shape[1], layer.weight.data.shape[2],
                         layer.weight.data.shape[3]))

                    channel_count = 0
                    for j in range(layer.weight.data.shape[0]):
                        if j not in prune_indices:
                            final_data[channel_count, :, :, :] = layer.weight.data[j]
                            channel_count += 1

                    final_data.to(device)
                    layer.out_channels = final_channel_num
                    layer.weight.data = final_data

                if isinstance(layer, nn.BatchNorm2d):
                    final_channel_num = layer.weight.data.shape[0] - prune_num
                    device = layer.weight.data.device
                    final_weight = torch.ones(final_channel_num)
                    final_bias = torch.ones(final_channel_num)
                    new_running_mean = torch.ones(final_channel_num)
                    new_running_var = torch.zeros(final_channel_num)

                    channel_count = 0
                    for j in range(layer.weight.data.shape[0]):
                        if j not in prune_indices:
                            final_weight[channel_count] = layer.weight.data[j]
                            final_bias[channel_count] = layer.bias.data[j]
                            new_running_mean[channel_count] = layer.running_mean[j]
                            new_running_var[channel_count] = layer.running_var[j]
                            channel_count += 1

                    layer.num_features = final_channel_num
                    final_weight.to(device)
                    final_bias.to(device)
                    new_running_mean.to(device)
                    new_running_var.to(device)
                    layer.weight.data = final_weight
                    layer.bias.data = final_bias
                    layer.register_buffer('running_mean', new_running_mean)
                    layer.register_buffer('running_var', new_running_var)
                    layer.running_mean = new_running_mean
                    layer.running_var = new_running_var

            if isinstance(post_layer, nn.Conv2d):
                next_conv_weight = torch.ones((post_layer.weight.data.shape[0], final_channel_num, post_layer.weight.data.shape[2], post_layer.weight.data.shape[3]))
                device = post_layer.weight.data.device

                channel_count = 0
                for j in range(post_layer.weight.data.shape[1]):
                    if j not in prune_indices:
                        next_conv_weight[:, channel_count, :, :] = post_layer.weight.data[:, j, :, :]
                        channel_count += 1

                next_conv_weight.to(device)
                post_layer.in_channels = final_channel_num
                post_layer.weight.data = next_conv_weight
            if isinstance(post_layer, nn.Linear):
                new_linear_weight = torch.ones((post_layer.out_features, final_channel_num))
                device = post_layer.weight.data.device

                channel_count = 0
                for j in range(post_layer.weight.data.shape[1]):
                    if j not in prune_indices:
                        new_linear_weight[:,channel_count] = post_layer.weight.data[:,j]
                        channel_count += 1

                new_linear_weight.to(device)
                post_layer.in_channels = final_channel_num
                post_layer.weight.data = new_linear_weight
            i += 1


    def compute_prune(self):
        self.prune_num = 0

        for i in range(len(self.masks)):
            self.prune_num += torch.sum(self.masks[i] == 0).item()

        for j in range(len(self.masks)):
            self.grow_num += torch.sum(self.masks[j] == 1).item()

        self.all_num = 0
        for s in range(len(self.masks)):
            self.all_num += self.masks[s].numel()

if __name__ == '__main__':
    # init all config settings
    config = init_config()

    mode = config['mode']

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
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    # load dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False if config['is_distributed'] or train_sampler is not None else True,
        sampler=train_sampler,
        num_workers=config['workers'],
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=test_sampler,
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

    if mode == 'load':
        model.load_state_dict(torch.load(config['model_path']))

    if global_rank == 0:
        logger.debug('\n' + str(model))

    model.to(device)

    idx_arr = config['pruning_nodes']
    pruningManager = PruningManager(model,idx_arr)

    # calculate number of parameters
    if global_rank == 0:
        n_parameters = count_parameters(model, trainable=True)
        logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

    if config['is_distributed']:
        model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    # init optimzer
    if config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'],
                              weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'].lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        if global_rank == 0:
            logger.warning(f"Received unrecognized optimizer {config['optimizer']}, set default Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # init scheduler
    if config['scheduler'].lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif config['scheduler'].lower() == 'linear':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config["epochs"] * 0.25), gamma=0.1)
    elif config['scheduler'].lower() == 'warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(config["epochs"] * 0.1), T_mult=2)
    else:
        if global_rank == 0:
            logger.warning(
                f"Received unrecognized scheduler {config['scheduler']}, set default ConsineAnnealing Scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_acc = 0.
    for epoch in range(1, config['epochs'] + 1):
        model.train()

        pruningManager.train()

        if config['is_distributed']:
            train_sampler.set_epoch(epoch)

        top1_meter, loss_meter = AverageMeter(), AverageMeter()
        # customize progress bar for train loader
        loader = tqdm(train_loader, unit='batch', ncols=80, desc='Train: ') if global_rank == 0 else train_loader
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            # default data shape (B, T, input_size) -> (T, B, input_size)
            inputs = inputs.transpose(0, 1)

            outputs = model(inputs)
            acc1 = accuracy(outputs, targets, topk=(1,))[0]

            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()


            top1_meter.update(acc1.item(), targets.numel())
            loss_meter.update(loss.item(), targets.numel())

        if not config['is_distributed'] or dist.get_rank() == 0:
            model.eval()

            pruningManager.eval()

            top1_meter, loss_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                    # default data shape (B, T, input_size) -> (T, B, input_size)
                    inputs = inputs.transpose(0, 1)

                    outputs = model(inputs)
                    acc1 = accuracy(outputs, targets, topk=(1,))[0]
                    loss = criterion(outputs, targets.long())

                    loss_meter.update(loss.item(), targets.numel())
                    top1_meter.update(acc1.item(), targets.numel())

            test_acc = top1_meter.avg

            logger.info(
                f"[Epoch {epoch:3d}/{config['epochs']}] Train Loss: {loss_meter.avg:.4f}, Train Acc: {top1_meter.avg:.2f}%; Test Loss: {loss_meter.avg:.4f}, Test Acc: {test_acc:.2f}%")
            if test_acc > best_acc:
                ensure_dir(config['model_dir'])

                best_acc = test_acc
                logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
                torch.save(
                    model.module.state_dict() if config['is_distributed'] else model.state_dict(),
                    os.path.join(config['model_dir'],
                                 f'best_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt')
                )

        scheduler.step()

        pruningManager.calculate_masks(model)
        pruningManager.dummy_pruning()
        pruningManager.compute_prune()

        if epoch == config['epochs']:
            pruningManager.structured_pruning()
            model.to(device)
            model.eval()

            pruningManager.eval()

            top1_meter, loss_meter = AverageMeter(), AverageMeter()
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                    # default data shape (B, T, input_size) -> (T, B, input_size)
                    inputs = inputs.transpose(0, 1)

                    outputs = model(inputs)
                    acc1 = accuracy(outputs, targets, topk=(1,))[0]
                    loss = criterion(outputs, targets.long())

                    loss_meter.update(loss.item(), targets.numel())
                    top1_meter.update(acc1.item(), targets.numel())

            test_acc = top1_meter.avg

            logger.info(
                f"[Epoch {epoch:3d}/{config['epochs']}] Train Loss: {loss_meter.avg:.4f}, Train Acc: {top1_meter.avg:.2f}%; Test Loss: {loss_meter.avg:.4f}, Test Acc: {test_acc:.2f}%")

            pruningManager.remove_hooks()
            torch.save(model,
                       os.path.join(config['model_dir'],f'pruned_{config["model"].lower()}_{config["neuron_type"].lower()}_{config["dataset_name"].lower()}_{config["seed"]}.pt'))


    # recycle all process
    if config['is_distributed']:
        dist.destroy_process_group()
