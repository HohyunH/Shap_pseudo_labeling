import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import sys
import random
import numpy as np
import pandas as pd


class BinaryDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        return text, label

def class_same(train_df, class_num):
  zero_df = train_df[train_df['label']==0][:class_num]
  one_df = train_df[train_df['label']==1][:class_num]
  total_df = pd.concat([zero_df, one_df])
  total_df=total_df.sample(frac=1).reset_index(drop=True)

  return total_df

if __name__=="__main__":
    torch.manual_seed(777)
    random.seed(777)
    np.random.seed(777)

    yelp = pd.read_csv('./yelp.csv')
    yelp['label'] = yelp['rating'] - 1
    del yelp['rating']

    train_df = yelp[:1000]
    test_df = yelp[1000:-1000]
    val_df = yelp[-1000:]

    train_df = class_same(train_df, 100)

    nsmc_train_dataset = BinaryDataset(train_df)
    train_loader = DataLoader(nsmc_train_dataset, batch_size=2, shuffle=True, num_workers=2)

    nsmc_eval_dataset = BinaryDataset(val_df)
    eval_loader = DataLoader(nsmc_eval_dataset, batch_size=2, shuffle=False, num_workers=2)

    for _ in train_loader:
        print(_)
        sys.exit(0)