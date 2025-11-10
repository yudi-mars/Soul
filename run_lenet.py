import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from soul.model import *
from soul.neuron import *
from soul.utils import *

class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.T = config['time_step']

        C, H, W = config['input_channels'], config['input_height'], config['input_width']

        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=0),
            nn.ReLU(True),
        )

        H = (H - 4) // 2
        W = (W - 4) // 2
        H = (H - 4) // 2
        W = (W - 4) // 2
        H -= 4
        W -= 4

        self.fc = nn.Sequential(
            nn.Linear(96 * H * W, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)

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
logger = setup_logger(os.path.join(log_path, f'record-{get_local_time()}.log'), default_level=config['state'])

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
        torch.save(
            model.state_dict(), 
            os.path.join(config['model_dir'], f'best_lenet_ann_{config["dataset_name"].lower()}_{config["seed"]}.pt')
        )

    scheduler.step()
