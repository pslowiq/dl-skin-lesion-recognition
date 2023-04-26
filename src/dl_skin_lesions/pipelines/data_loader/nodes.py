"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
import pandas as pd

from kedro.io import PartitionedDataSet
from dl_skin_lesions.extras.datasets.image_dataset import ImageDataset
import torch
from torch.utils.data import random_split

def calculate_train_weights(dataset, num_classes):

    train_weights = {i : 0 for i in range(num_classes)}

    for _ , label in dataset:
        train_weights[label] += 1

    for label in train_weights.keys():
        train_weights[label] = 1. - train_weights[label]/len(dataset)

    return [train_weights[i] for i in range(num_classes)]

def create_torch_dataset(images : PartitionedDataSet, csv : pd.DataFrame, loader_params):
    dataset = ImageDataset(images, csv, image_size = loader_params['image_size'])

    return dataset

def split_data(dataset : ImageDataset, loader_params):

    train_test_split = loader_params['train_test_split']
    seed = loader_params['random_seed']
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(dataset, train_test_split, generator=generator)

    train_weights = calculate_train_weights(train_dataset, loader_params['num_classes'])

    return train_dataset, train_weights, test_dataset