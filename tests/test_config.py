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
    assert(config.precision == PRECISIONS[precision])


@pytest.mark.parametrize('backend', ['np', 'cu'])
def test_config_set_backend(backend):
    config.set_backend(backend)
    assert(config.backend == backend)


@pytest.mark.parametrize('zbase', [0, 1])
def test_config_set_zernike_base(zbase):
    config.set_zernike_base(zbase)
    assert(config.zernike_base == zbase)
