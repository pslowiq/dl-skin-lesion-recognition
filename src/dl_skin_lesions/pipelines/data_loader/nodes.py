"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.io import PartitionedDataSet
import torch
from torch.utils.data import random_split, Dataset

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

# different name because of csv
def load_data_to_np(images : PartitionedDataSet, csv : pd.DataFrame, loader_params):

    image_size = loader_params['image_size']
    csv['label'] = pd.Categorical(csv['dx']).codes

    x = np.stack([data_fun().resize(image_size) for _, data_fun in images.items()])
    y = np.stack([csv[csv['image_id'] == id]['label'].values[0] for id, _ in images.items()])

    return (x, y), csv

def split_data(dataset : Tuple[np.array, np.array], loader_params):
    train_test_split = loader_params['train_test_split']

    x, y = dataset
    dataset = CustomIterable(x, y)
    train_dataset, test_dataset = random_split(dataset, train_test_split, generator=torch.Generator().manual_seed(42))
    return train_dataset, test_dataset