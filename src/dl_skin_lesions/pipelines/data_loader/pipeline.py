"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.5
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
        node(func = load_data_to_np, inputs = ["HAM10000", "HAM10000_metadata"], outputs = ["HAM10000_np", "HAM10000_metadata_with_categories"])
        ]
    )
