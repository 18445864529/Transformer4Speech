import torch
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from transformer import Transformer
from utils import yaml_load
from dataset import BaseDataLoader


def get_preprocess_args():
    parser = argparse.ArgumentParser(description='Argument Parser for main function.')
    parser.add_argument('--data_path', default='../../Dataset/LibriSpeech', type=str,
                        help='Path to raw LibriSpeech dataset')
    parser.add_argument('--path_save', default='./log/save_net.pkl', type=str,
                        help='Path to raw LibriSpeech dataset')
    args = parser.parse_args()
    return args


args = get_preprocess_args()
config = yaml_load('config.yaml')
feature_dir = config.feature_directory
feature_dim = config.feature_dim

data_loader = BaseDataLoader(config)
train_loader = data_loader.get_loader('train')

model = Transformer(idim=feature_dim, odim=config.dic_dim, nlayer=6, ahead=2, adim=10).cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.003)
model.train()

print("Starting Training Loop...")
for ep in range(config.epoch):
    pbar = tqdm(enumerate(train_loader), leave=True)
    for i, (x, x_len, y) in pbar:
        optim.zero_grad()
        x, x_len, y = x.cuda(), x_len.cuda(), y.cuda()
        loss = model(x, x_len, y)
        loss.backward()
        optim.step()
        # print(f'{loss.item():.2f}')
        pbar.set_description(f'epoch:{ep} | Total samples: {len(train_loader)} |')
        pbar.set_postfix(loss=loss.item())

torch.save(model, './save_net.pkl')
