import sys
import os
import torch
import multiprocessing
import numpy as np
import pandas as pd
from torch.utils import data
from utiles import yaml_load
from tqdm import tqdm
from ast import literal_eval

config = yaml_load('config.yaml')


class T4SDataset(data.Dataset):
    def __init__(self, config, df):
        self.config = config
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature_path = self.df.iloc[idx].file_path
        x = np.load(os.path.join(config.feature_directory, str(feature_path)))
        y_all = self.df['token'].values.tolist()
        y = literal_eval(y_all[idx])
        return x, y  # x: (T, F) y: list[0,0,0]


class BaseDataLoader:
    def __init__(self, config):
        self.config = config
        self.shuffle = config.shuffle
        self.batch_size = config.batch_size
        self.train_df = pd.read_csv(config.token_path)

    @staticmethod
    def collate_fn(batch):
        x_list = []
        x_len_list = []
        y_list = []
        for index, (x, y) in enumerate(batch):
            x_list.append(torch.tensor(x).float())
            x_len_list.append(torch.tensor(x).size(0))
            y_list.append(torch.tensor(y).long())
        padded_x = torch.nn.utils.rnn.pad_sequence(x_list, batch_first=True)
        padded_y = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True, padding_value=-1)
        x_len = torch.tensor(x_len_list)
        # print(padded_x)
        # y = torch.stack(y_list, dim=0)
        return padded_x, x_len, padded_y

    def get_loader(self, for_what):
        train_dataset = T4SDataset(self.config, self.train_df)
        dataset = eval(f'{for_what}_dataset')
        if for_what == 'test' or 'valid':
            self.shuffle = False
        return data.DataLoader(dataset, batch_size=self.batch_size,
                               num_workers=0,
                               shuffle=self.shuffle,
                               collate_fn=self.collate_fn)


if __name__ == '__main__':
    dl = BaseDataLoader(config)
    tl = dl.get_loader('train')
    print(tl)
    for i, (x, x_len, y) in enumerate(tl):
        print(i)
        print(x.shape)  # torch.Size([batch, 2974, 160])
        print(x_len)
        print(y.size())
        if i == 1:
            break
