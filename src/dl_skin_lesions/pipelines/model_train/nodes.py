"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.18.6
"""


from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pytorch_lightning as pl

from kedro.framework.project import settings
from kedro.config import ConfigLoader
cfg = ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))
params = cfg['parameters']



class CustomIterable(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        img, label = self.x[idx], self.y[idx]
        img_tensor = torch.tensor(img, dtype=torch.float32) / 255
        img_tensor = img_tensor.permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.int64)
        return img_tensor, label
    
def train_model(model, dataset):
    x, y = dataset
    data_loader = DataLoader(CustomIterable(x, y), batch_size=32)

    x = torch.tensor(x, dtype=torch.float32)
    t = pl.Trainer(max_epochs=params['epochs'])
    t.fit(model=model, train_dataloaders=data_loader)
    return model