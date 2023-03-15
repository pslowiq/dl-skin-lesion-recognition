"""
This is a boilerplate pipeline 'model_create'
generated using Kedro 0.18.6
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import lightning as pl

import torchmetrics

from kedro.framework.project import settings
from kedro.config import ConfigLoader

cfg = ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))
params = cfg['parameters']

class BasicCNN(pl.LightningModule):
    def __init__(self, channels_out, kernel_size):
        image_shape = params['image_size']

        super().__init__()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=7)
        self.cnn1 = nn.Conv2d(3, channels_out, kernel_size)
        self.dense1 = nn.Linear((image_shape[0] - kernel_size+1) * (image_shape[1] - kernel_size+1) * channels_out, 7)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = nn.CrossEntropyLoss()(x,y)
        
        self.log('train acc', self.train_acc(x, y))
        self.log('train loss', loss)

        return loss
    
    def forward(self, x):
        x = self.cnn1(x)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)
        x = self.dense1(x)
        x = nn.Softmax(dim = -1)(x)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        return optimizer


def create_model():
    return BasicCNN(params['channels_out'], params['kernel_size'])



