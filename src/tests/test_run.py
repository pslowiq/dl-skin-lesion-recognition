"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from dvc.api import DVCFileSystem
import numpy as np

@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="dl_skin_lesions",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality



class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == Path.cwd()
    def test_image_load(self, config_loader):
        from dl_skin_lesions.pipelines.data_loader.nodes import load_image_from_fs
        params = config_loader['parameters']
        fs = DVCFileSystem(url = params['repo_url'], rev = 'main')
        img = load_image_from_fs([x['name'] for x in fs.listdir('/data/01_raw/HAM10000/HAM10000_images')][0], fs)

        assert type(img) == np.ndarray
