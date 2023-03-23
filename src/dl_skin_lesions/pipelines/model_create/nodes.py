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

class BasicCNN(pl.LightningModule):
    def __init__(self, channels_out, kernel_size, image_shape, num_classes, learning_rate):

        self.learning_rate = learning_rate
        super().__init__()
        self.multiclass_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.cnn1 = nn.Conv2d(3, channels_out, kernel_size)
        self.dense1 = nn.Linear((image_shape[0] - kernel_size+1) * (image_shape[1] - kernel_size+1), num_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = nn.CrossEntropyLoss()(x,y)
        
        self.log('train acc', self.multiclass_accuracy(x, y), on_step = False, on_epoch = True)
        self.log('train loss', loss, on_step = False, on_epoch = True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.forward(x)
        loss = nn.CrossEntropyLoss()(x,y)
        
        self.log('validation acc', self.multiclass_accuracy(x, y), on_step = False, on_epoch = True)
        self.log('validation loss', loss, on_step = False, on_epoch = True)

        return loss
    
    def forward(self, x):
        x = self.cnn1(x)
        x = x.mean(dim = 1)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)
        x = self.dense1(x)
        #x = nn.Softmax(dim = -1)(x)
        return x


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def create_model(create_params, loader_params):
    return BasicCNN(create_params['channels_out'], create_params['kernel_size'], loader_params['image_size']
                    , loader_params['num_classes'], create_params['learning_rate'])



