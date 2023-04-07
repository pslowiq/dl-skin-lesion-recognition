"""
This is a boilerplate pipeline 'model_create'
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func = create_model, inputs = ['params:create_params', 'params:loader_params', 'train_weights'], outputs = 'lightning_model')
    ])
