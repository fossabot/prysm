''' unit tests for the mathops submodule.
'''
import pytest

import numpy as np

from prysm import mathops

pyculib = pytest.importorskip('pyculib')
np.random.seed(1234)
TEST_ARR_SIZE = 32


@pytest.fixture
def sample_data_2d():
    return np.random.rand(TEST_ARR_SIZE, TEST_ARR_SIZE)


@pytest.mark.skipif(not pyculib, reason='pyculib not installed / no CUDA support here.')
def test_cufft_returns_array_same_size(sample_data_2d):
    result = mathops.cu_fft2(sample_data_2d)
    assert result.shape == sample_data_2d.shape


@pytest.mark.skipif(not pyculib, reason='pyculib not installed / no CUDA support here.')
def test_cuifft_returns_array_same_size(sample_data_2d):
    result = mathops.cu_ifft2(sample_data_2d)
    assert result.shape == sample_data_2d.shape


def test_mathops_handles_own_jit_and_vectorize_definitions():
    from importlib import reload
    from unittest import mock

    class FakeNumba():
        __version__ = '0.35.0'

    with mock.patch.dict('sys.modules', {'numba': FakeNumba()}):
        reload(mathops)  # may have side effect of disabling numba for downstream tests.

        def foo():
            pass

        foo_jit = mathops.jit(foo)
        foo_vec = mathops.vectorize(foo)

        assert foo_jit == foo
        assert foo_vec == foo


# below here, tests purely for function not accuracy
def test_fft2(sample_data_2d):
    result = mathops.fft2(sample_data_2d)
    assert type(result) is np.ndarray


def test_ifft2(sample_data_2d):
    result = mathops.ifft2(sample_data_2d)
    assert type(result) is np.ndarray


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cast_array(dtype):
    arr = np.ones((TEST_ARR_SIZE, TEST_ARR_SIZE), dtype=dtype)
    assert mathops.cast_array(arr).all()
