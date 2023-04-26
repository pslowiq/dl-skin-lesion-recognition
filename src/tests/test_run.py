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
    """
    Tests if the configurations can be acquired from the source path.
    """
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    """
    Tests if the full Kedro context loads properly.
    """
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
