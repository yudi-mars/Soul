import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from soul.model import *
from soul.neuron import *
from soul.utils import *

class BinaryActivationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()   # >0 -> +1, <=0 -> -1

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        grad_input = grad_output.clone()
        grad_input = grad_input * (grad_output.abs() <= 1).float()
        return grad_input


class BinaryActivation(nn.Module):
    """可以像 nn.ReLU 一样在 nn.Sequential 中使用"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return BinaryActivationFn.apply(x)
    
class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.app = config['application']

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        if self.app in ['vision', 'wireless']:
            C, H, W = config['input_channels'], config['input_height'], config['input_width']
        elif self.app in ['motion']:
            H, W = config['input_channels'], config['input_dim']
            C = 1
        elif self.app in ['acoustic']:
            H, W = config['input_dim'], config['input_channels']
            C = 1
        else:
            raise ValueError(self.app)
        
        conv_kernel_size = (1, 5) if self.app == 'motion' else 5
        pool_kernel_size = (1, 2) if self.app == 'motion' else 2
        pool_stride_size = (1, 2) if self.app == 'motion' else 2

        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=conv_kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(32),
            # BinaryActivation(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride_size),
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(64),
            # BinaryActivation(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride_size),
            nn.Conv2d(64, 96, kernel_size=conv_kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(96),
            # BinaryActivation(),
            nn.ReLU(True),
        )

        if self.app in ['wireless', 'vision', 'acoustic']:
            H = (H - 4) // 2
            W = (W - 4) // 2
            H = (H - 4) // 2
            W = (W - 4) // 2
            H -= 4
            W -= 4
        else:
            W = (W - 4) // 2
            W = (W - 4) // 2
            W -= 4

        dim = 96 * H * W

        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            # BinaryActivation(),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)

        if self.app in ['motion', 'acoustic']:
            x = x.unsqueeze(1) # (B, C, W) -> (B, 1, C, W)

        x = self.encoder(x)
        x = x.flatten(1) # (B, C, H, W) -> (B, C*H*W)

        out = self.fc(x) # (B, C*H*W) -> (B, num_classes)

        return out

# init all config settings
config = init_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_path = os.path.join(
    config['log_dir'], 
    config['dataset_name'].lower(), 
    config['model'].lower(), 
    'ann',
)
ensure_dir(log_path)
if config['noise_intensity'] > 0.:
    logger = setup_logger(os.path.join(log_path, f'noise-record-{get_local_time()}.log'), default_level=config['state'])
else:
    logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])

if config['dataset_name'].lower() in ['dvsgesture', 'ssc', 'shd', 'cifar10dvs']:
    config['time_step'] = 10 

# report all configuration
for k, v in sorted(config.items()):
    logger.debug(f'{k} = {v}')

# reproducibility
logger.info(f'Reproducibility with random seed {config["seed"]}')
init_seed(config["seed"])
logger.info('=' * 50)

logger.info('Load data...')
train_dataset, test_dataset = load_dataset(config)

# load dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=config['batch_size'], 
    shuffle= True,
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

logger.info(f'Load ANN model: LeNet...')
logger.info(f'#Training Samples: {len(train_dataset)}; #Test Samples: {len(test_dataset)}')

model = LeNet(config)
model.to(device)

# calculate number of parameters
n_parameters = count_parameters(model, trainable=True) 
logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")

criterion = nn.CrossEntropyLoss()
# init optimzer
if config['optimizer'].lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'], weight_decay=config['weight_decay'])
elif config['optimizer'].lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
elif config['optimizer'].lower() == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
elif config['optimizer'].lower() == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
else:
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
    logger.warning(f"Received unrecognized scheduler {config['scheduler']}, set default ConsineAnnealing Scheduler")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])


best_acc = 0.
for epoch in range(1, config['epochs'] + 1):
    model.train()
    
    train_top1_meter, train_loss_meter = AverageMeter(), AverageMeter()
    # customize progress bar for train loader
    loader = tqdm(train_loader, unit='batch', ncols=80, desc='Train: ')
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        # default data shape (B, T, input_size) -> (T, B, input_size)
        inputs = inputs.transpose(0, 1)

        # apply noise if needed
        if config['noise_phase'] == 'train' and config['noise_intensity'] > 0.:
            inputs = noising_map[config['application']][config['noise_type']](inputs, config['noise_intensity'])

        outputs = model(inputs)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_top1_meter.update(acc1.item(), targets.numel())
        train_loss_meter.update(loss.item(), targets.numel())

    train_acc = train_top1_meter.avg
    train_loss = train_loss_meter.avg

    
    model.eval()

    test_top1_meter, test_loss_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # default data shape (B, T, input_size) -> (T, B, input_size)
            inputs = inputs.transpose(0, 1)

            # apply noise if needed
            if config['noise_phase'] == 'test' and config['noise_intensity'] > 0.:
                inputs = noising_map[config['application']][config['noise_type']](inputs, config['noise_intensity'])

            outputs = model(inputs)
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            loss = criterion(outputs, targets)

            test_loss_meter.update(loss.item(), targets.numel())
            test_top1_meter.update(acc1.item(), targets.numel())

    test_acc = test_top1_meter.avg
    test_loss = test_loss_meter.avg

    logger.info(f"[Epoch {epoch:3d}/{config['epochs']}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%; Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    if test_acc > best_acc:
        ensure_dir(config['model_dir'])

        best_acc = test_acc
        logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
        if not config['noise_intensity'] > 0.:
            torch.save(
                model.state_dict(), 
                os.path.join(config['model_dir'], f'best_lenet_ann_{config["dataset_name"].lower()}_{config["seed"]}.pt')
            )

    scheduler.step()
