"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
        node(func = create_torch_dataset, inputs = ["HAM10000", "HAM10000_metadata", "params:loader_params"], outputs = "dataset"),
        node(func = split_data, inputs = ["dataset","params:loader_params"], outputs = ["train_dataset", "train_weights", "test_dataset"])
        ]
    )
