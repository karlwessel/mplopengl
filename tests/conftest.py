import matplotlib as mpl
from matplotlib.testing.conftest import (mpl_test_settings,
                                         mpl_image_comparison_parameters,
                                         pytest_configure, pytest_unconfigure,
                                         pd)
import pytest
@pytest.fixture(autouse=True)
def test_settings():
    mpl.rcParams['savefig.dpi'] = 80.0