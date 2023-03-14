"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from pathlib import Path
import numpy as np
import pandas as pd

from kedro.framework.project import settings
from kedro.config import ConfigLoader
cfg = ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))

def load_data_to_np(images, csv):

    csv['label'] = pd.Categorical(csv['dx']).codes

    x = np.stack([data_fun().resize(cfg['parameters']['image_size']) for id, data_fun in images.items()])
    y = np.stack([csv[csv['image_id'] == id]['label'].values[0] for id, data_fun in images.items()])

    return (x, y), csv
