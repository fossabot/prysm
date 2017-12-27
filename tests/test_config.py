''' Unit tests verifying the functionality of the global prysm config.
'''
import pytest

import numpy as np

from prysm import config

PRECISIONS = {
    32: np.float32,
    64: np.float64,
}


@pytest.mark.parametrize('precision', [32, 64])
def test_config_set_precision(precision):
    config.set_precision(precision)
    assert config.precision == PRECISIONS[precision]


# must make certain the backend is set to numpy last to avoid cuda errors for rest of test suite
@pytest.mark.parametrize('backend', ['cu', 'np'])
def test_config_set_backend(backend):
    config.set_backend(backend)
    assert config.backend == backend


def test__force_testenv_backend_numpy():
    config.set_backend('np')
    assert config


@pytest.mark.parametrize('zbase', [0, 1])
def test_config_set_zernike_base(zbase):
    config.set_zernike_base(zbase)
    assert config.zernike_base == zbase
