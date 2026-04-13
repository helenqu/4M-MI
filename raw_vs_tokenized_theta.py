import torch
import numpy as np
import yaml
from pathlib import Path
import copy
from functools import partial

from fourm.vq.vqvae import VQVAE
from fourm.data.mi_dataset import MIDataset
from fourm.data.dataset_utils import split_mi_data_paths
from run_training_vqvae import get_model

from types import SimpleNamespace

# input shape: [3, 32, 32]
# embedding shape: [32, 8, 8]
class ThetaPredictorFromTokens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [32, 8, 8]
        self.conv1 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device, tokenizer=None, return_tokens=False):
    model.to(device)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if tokenizer is not None:
            if return_tokens:
                inputs, _, _ = tokenizer.encode(batch[domain].to(device))
            else:
                inputs, _ = tokenizer(batch[domain].to(device))
        else:
            inputs = batch[domain].to(device)
        target = batch['theta'].to(device)
        optimizer.zero_grad()
        output = model(inputs).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}", flush=True)

def evaluate(model, dataloader, criterion, device, tokenizer=None, return_tokens=False):
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if tokenizer is not None:
                if return_tokens:
                    inputs, _, _ = tokenizer.encode(batch[domain].to(device))
                else:
                    inputs, _ = tokenizer(batch[domain].to(device))
            else:
                inputs = batch[domain].to(device)
            target = batch['theta'].to(device)
            output = model(inputs).squeeze()
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def collate_fn(batch, theta_max=None, theta_min=None, theta_std=None):
    rgb1 = torch.stack([torch.tensor(b['rgb1']) for b in batch])
    rgb2 = torch.stack([torch.tensor(b['rgb2']) for b in batch])
    theta = torch.stack([torch.tensor(b['theta']) for b in batch])
    have_theta_stats = theta_max is not None and theta_min is not None and theta_std is not None
    return {
        'rgb1': rgb1,
        'rgb2': rgb2,
        'theta': (theta - theta_min) / (theta_max - theta_min) if have_theta_stats else theta
    }

# CONFIG
domain = "rgb2"
data = "case2_rho0999"
lr = 1e-4
predict_on_tokens = True

with open(f"/mnt/home/hqu10/ml-4m/cfgs/default/tokenization/vqvae/rgb/mi_{domain}.yaml", "r") as f:
    args = yaml.safe_load(f)
args['full_ckpt'] = None
args['encoder_ckpt'] = None
args['mask_value'] = args.get('mask_value', None)
args['input_size_enc'] = args.get('input_size_enc', None)
args['input_size_dec'] = args.get('input_size_dec', None)
args['num_codebooks'] = 1
args['norm_latents'] = False
args['distributed'] = False
# args['quantizer_ema_decay'] = args.get('quantizer_ema_decay', 0.8)
args['code_replacement_policy'] = args.get('code_replacement_policy', 'batch_random')
# args['kmeans_init'] = False
args['freeze_enc'] = False
args['out_conv'] = False

args = SimpleNamespace(**args)

tokenizer = get_model(args, "cuda")
ckpt = torch.load(f"/mnt/ceph/users/hqu10/mi_outputs/{domain}_tokenizer_{data}/checkpoint-final.pth", weights_only=False)
tokenizer.load_state_dict(ckpt['model'])
tokenizer.eval()
tokenizer.to("cuda")

parent_data_dir = Path("/mnt/home/hqu10/ceph/mi_datasets")
train_paths, val_paths, test_paths = split_mi_data_paths(ckpt['args'].data_path)
print(f"loading dataset from {ckpt['args'].data_path}")
train_dataset = MIDataset(train_paths)
val_dataset = MIDataset(val_paths)
theta_max = np.max(train_dataset.thetas)
theta_min = np.min(train_dataset.thetas)
theta_std = np.std(train_dataset.thetas)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=128, 
    collate_fn=partial(collate_fn, theta_max=theta_max, theta_min=theta_min, theta_std=theta_std), 
    shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=128, 
    collate_fn=partial(collate_fn, theta_max=theta_max, theta_min=theta_min, theta_std=theta_std), 
    shuffle=False
)

cifar10_trained = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
cifar10_trained.fc = torch.nn.Linear(64, 1, bias=False)
raw_predictor = cifar10_trained

tokenized_predictor = ThetaPredictorFromTokens().to("cuda") if predict_on_tokens else copy.deepcopy(cifar10_trained)

raw_optimizer = torch.optim.Adam(raw_predictor.parameters(), lr=lr)
tokenized_optimizer = torch.optim.Adam(tokenized_predictor.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

print("Training raw predictor", flush=True)
for epoch in range(4):
    train_one_epoch(epoch, raw_predictor, train_dataloader, raw_optimizer, criterion, "cuda")
    raw_loss = evaluate(raw_predictor, train_dataloader, criterion, "cuda")
    print(f"epoch {epoch}, raw val loss: {raw_loss}", flush=True)

# print("Training tokenized predictor")
# for epoch in range(5):
#     train_one_epoch(epoch, tokenized_predictor, train_dataloader, tokenized_optimizer, criterion, "cuda", tokenizer=tokenizer, return_tokens=predict_on_tokens)
#     tokenized_loss = evaluate(tokenized_predictor, train_dataloader, criterion, "cuda", tokenizer=tokenizer, return_tokens=predict_on_tokens)
#     print(f"epoch {epoch}, tokenized val loss: {tokenized_loss}", flush=True)
