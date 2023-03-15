"""
This is a boilerplate pipeline 'model_train'
generated using Kedro 0.18.6
"""


from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch import loggers as pl_loggers

from kedro.framework.project import settings
from kedro.config import ConfigLoader
cfg = ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))
params = cfg['parameters']


def train_model(model, dataset):

    data_loader = DataLoader(dataset, batch_size=params['batch_size'], num_workers=12)
    tb_logger = pl_loggers.CSVLogger(save_dir = Path.cwd() / "logs/")

    t = pl.Trainer(max_epochs=params['epochs'], logger=tb_logger)
    t.fit(model=model, train_dataloaders=data_loader)
    
    return model