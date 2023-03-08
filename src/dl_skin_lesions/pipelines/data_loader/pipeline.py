"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from pathlib import Path

from kedro.pipeline import Pipeline, node, pipeline
from kedro.extras.datasets.text import TextDataSet

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.extras.datasets.text import TextDataSet
from kedro.io import DataCatalog
from .nodes import *

cfg = ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        []
    )
