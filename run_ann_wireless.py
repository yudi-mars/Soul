import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from soul.model import *
from soul.neuron import *
from soul.utils import *

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 1, patch_size_w = 9, patch_size_h = 25, emb_size = 9*25, img_size = 342*500):
        self.patch_size_w = patch_size_w
        self.patch_size_h = patch_size_h
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.position = nn.Parameter(torch.randn(int(img_size/emb_size) + 1, emb_size))  

    def forward(self, x):
        x = x.mean(0)  # (T, B, C, H, W) -> (B, C, H, W)
        b, C, H, W = x.shape
        x = x.view(-1, 1, C * H, W)
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 225, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 225,
                 drop_p = 0.5,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 1, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, num_classes))

class ViT(nn.Sequential):
    def __init__(self, config):
        self.num_classes = config['num_classes']
        self.A, self.S, seq_len = config['input_channels'], config['input_height'], config['input_width']

        if config['dataset_name'] == 'fihumanid':
            patch_size_w, patch_size_h = 9, 25
            num_heads = 5
        elif config['dataset_name'] == 'uthar':
            patch_size_w, patch_size_h = 50, 18
            num_heads = 5
        elif config['dataset_name'] == 'aril':
            patch_size_w, patch_size_h = 13, 16
            num_heads = 4
        elif config['dataset_name'] == 'bullydetect':
            patch_size_w, patch_size_h = 8, 50
            num_heads = 5
        else:
            raise NotImplementedError(f'Dataset {config["dataset_name"]} not implemented yet for ViT with unknown patch size!')
        depth = 1
        in_channels = 1
        emb_size = patch_size_w * patch_size_h 

        super().__init__(
            PatchEmbedding(in_channels, patch_size_w, patch_size_h, emb_size, self.A * self.S * seq_len),
            TransformerEncoder(depth, emb_size=emb_size, num_heads=num_heads),
            ClassificationHead(emb_size, self.num_classes)
        )

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        C, H, W = config['input_channels'], config['input_height'], config['input_width']

        self.fc = nn.Sequential(
            nn.Linear(C * H * W, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # return to normal ann input shape
        x = x.mean(0)  # (T, B, C, H, W) -> (B, C, H, W)
        x = x.flatten(1)  # (B, C, H, W) -> (B, CHW)
        x = self.fc(x)  # (B, CHW) -> (B, num_classes)
        return x
    
class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        C, H, W = config['input_channels'], config['input_height'], config['input_width']

        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )

        H = (H - 4) // 2
        W = (W - 4) // 2
        H = (H - 4) // 2
        W = (W - 4) // 2
        H -= 4
        W -= 4

        dim = 96 * H * W
        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)

        x = self.encoder(x)
        x = x.flatten(1) # (B, C, H, W) -> (B, C*H*W)
        out = self.fc(x) # (B, C*H*W) -> (B, num_classes)

        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x
    
class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, C=3):
        super(ResNet, self).__init__()

        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(C, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
class RNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config['num_classes']
        self.C, self.H, self.W = config['input_channels'], config['input_height'], config['input_width']
        hidden_dim = 64

        self.rnn = nn.RNN(self.C * self.H, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        x = x.mean(0)  # (T, B, C, H, W) -> (B, C, H, W)
        x = x.view(-1, self.C * self.H, self.W)  # (B, C, H, W) -> (B, C * H, W)
        x = x.permute(2, 0, 1) # (B, C * H, W) -> (W, B, C * H)

        _, ht = self.rnn(x)
        outputs = self.fc(ht[-1])

        return outputs

class GRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.C, self.H, self.W = config['input_channels'], config['input_height'], config['input_width']
        hidden_dim = 64

        self.gru = nn.GRU(self.C * self.H, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)

        x = x.view(-1, self.C * self.H, self.W)  # (B, C, H, W) -> (B, C * H, W)
        x = x.permute(2, 0, 1) # (B, C * H, W) -> (W, B, C * H)

        _, ht = self.gru(x)
        outputs = self.fc(ht[-1])

        return outputs
    
class CNNGRUModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.C, self.H, self.W = config['input_channels'], config['input_height'], config['input_width']
        self.in_channels = self.C * self.H
        
        # Conv1d encoder layer
        layers = []
        prev_channels = self.in_channels
        for out_ch in [64, 128]:
            layers.append(nn.Conv1d(prev_channels, out_ch, kernel_size=3, padding=3 // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.3))
            prev_channels = out_ch

        self.cnn = nn.Sequential(*layers)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)
        B, A, S, T = x.shape
        x = x.view(B, A*S, T)

        x = self.cnn(x)  # (B, C', T)

        x = x.permute(0, 2, 1) # (B, T, C')
        out, _ = self.gru(x)
        outputs = self.fc(out[:, -1, :])

        return outputs

    
class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.C, self.H, self.W = config['input_channels'], config['input_height'], config['input_width']
        hidden_dim = 64

        self.lstm = nn.LSTM(self.C * self.H, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)
        x = x.view(-1, self.C * self.H, self.W)  # (B, C, H, W) -> (B, C * H, W)
        x = x.permute(2, 0, 1) # (B, C * H, W) -> (W, B, C * H)

        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])

        return outputs
    
class BiLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_classes = config['num_classes']
        self.C, self.H, self.W = config['input_channels'], config['input_height'], config['input_width']
        hidden_dim = 64

        self.lstm = nn.LSTM(self.C * self.H, hidden_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, self.num_classes)


    def forward(self, x):
        x = x.mean(0) # (T, B, C, H, W) -> (B, C, H, W)
        x = x.view(-1, self.C * self.H, self.W)  # (B, C, H, W) -> (B, C * H, W)
        x = x.permute(2, 0, 1) # (B, C * H, W) -> (W, B, C * H)

        _, (ht, ct) = self.lstm(x)
        outputs = self.fc(ht[-1])

        return outputs
    


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

logger.info(f'Load ANN model: {config["model"].lower()}...')
logger.info(f'#Training Samples: {len(train_dataset)}; #Test Samples: {len(test_dataset)}')

if config['model'].lower() == 'lenet':
    model = LeNet(config)
elif config['model'].lower() == 'mlp':
    model = MLP(config)
elif config['model'].lower() == 'resnet18':
    model = ResNet(Block, [2, 2, 2, 2], num_classes=config['num_classes'], C=config['input_channels'])
elif config['model'].lower() == 'resnet50':
    model = ResNet(Block, [3, 4, 6, 3], num_classes=config['num_classes'], C=config['input_channels'])
elif config['model'].lower() == 'resnet101':
    model = ResNet(Block, [3, 4, 23, 3], num_classes=config['num_classes'], C=config['input_channels'])
elif config['model'].lower() == 'rnn':
    model = RNNModel(config)
elif config['model'].lower() == 'gru':
    model = GRUModel(config)
elif config['model'].lower() == 'lstm':
    model = LSTMModel(config)
elif config['model'].lower() == 'bilstm':
    model = BiLSTMModel(config)
elif config['model'].lower() == 'cnngru':
    model = CNNGRUModel(config)
elif config['model'].lower() == 'vit':
    model = ViT(config)
else:
    raise NotImplementedError(f'Model {config["model"]} not implemented yet!')
model.to(device)

# calculate number of parameters
n_parameters = count_parameters(model, trainable=True) 
logger.info(f"Number of params for model {config['model']}: {n_parameters / 1e6:.2f} M")


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

best_acc = 0.
for epoch in range(1, config['epochs'] + 1):
    model.train()
    
    train_top1_meter, train_loss_meter = AverageMeter(), AverageMeter()
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
        if not config['noise_intensity'] > 0.:
            torch.save(
                model.state_dict(), 
                os.path.join(config['model_dir'], f'best_{config["model"].lower()}_ann_{config["dataset_name"].lower()}_{config["seed"]}.pt')
            )

    scheduler.step()

########################### inference process ###########################
# # calculate theoretical energy cost per sample inference
# best_model_path = os.path.join(config['model_dir'], f'best_{config["model"].lower()}_ann_{config["dataset_name"].lower()}_{config["seed"]}.pt')
# best_params = torch.load(
#     best_model_path, 
#     map_location='cpu'
# )
# model.load_state_dict(best_params)
# model.to(device)
# model.eval()

# # TODO 验证下ann-lenet推理 calculate theoretical energy cost per sample inference
# print('Counting FLOPs/SOPs for theoretical inference cost')
# total_flops, num_samples = 0, 0
# for inputs, _ in tqdm(test_loader, unit='batch', ncols=80, desc='Count OPs: '):
#     inputs = inputs.to(device, non_blocking=True)
#     # default data shape (B, T, input_size) -> (T, B, input_size)
#     inputs = inputs.transpose(0, 1)

#     num_samples += inputs.size(1)

#     flops = count_flops(model, inputs)

#     total_flops += flops

# avg_flops = total_flops / num_samples
# avg_energy_per_sample = avg_flops * config['e_mac']
# print(f"Average number of Operations (#OPs): {avg_flops /1e6:.2f} M, corresponding theoretical energy cost: {avg_energy_per_sample / 1e9:.2f} mJ")