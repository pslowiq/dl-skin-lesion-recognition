"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from pathlib import Path

from dvc.api import DVCFileSystem
import torchvision.io as tio
import cv2
import numpy as np

from kedro.framework.project import settings
from kedro.config import ConfigLoader
cfg = ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))

def load_image_from_fs(path : str, fs : DVCFileSystem) -> np.array:
    image_bytes = fs.open(path).read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np

def load_image_from_fsstr(path : str, fs_url : DVCFileSystem) -> np.array:
    fs = fs = DVCFileSystem(url = fs_url, rev = 'main')
    image_bytes = fs.open(path).read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np


def load_data(url):
    fs = DVCFileSystem(url = url, rev = 'main')
    #TODO
    