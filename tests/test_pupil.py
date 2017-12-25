''' Unit tests for pupil objects
'''
import pytest

from prysm import Pupil


@pytest.fixture
def p():
    return Pupil()


def test_create_pupil():
    p = Pupil()
    assert hasattr(p, 'wavelength')
    assert hasattr(p, 'epd')
    assert hasattr(p, 'sample_spacing')
    assert hasattr(p, 'samples')
    assert hasattr(p, 'opd_unit')
    assert hasattr(p, '_opd_unit')
    assert hasattr(p, '_opd_str')
    assert hasattr(p, 'phase')
    assert hasattr(p, 'fcn')
    assert hasattr(p, 'unit')
    assert hasattr(p, 'rho')
    assert hasattr(p, 'phi')
    assert hasattr(p, 'center')


def test_pupil_passes_valid_params():
    parameters = {
        'samples': 16,
        'epd': 128.2,
        'wavelength': 0.6328,
        'opd_unit': 'nm',
    }
    p = Pupil(**parameters)
    assert(p.samples == parameters['samples'])
    assert(p.epd == parameters['epd'])
    assert(p.wavelength == parameters['wavelength'])
    assert(p._opd_str == parameters['opd_unit'])
    assert(p._opd_unit == 'nanometers')  # make sure this is updated if the test is changed to a different unit


def test_pupil_has_zero_pv(p):
    assert(p.pv == pytest.approx(0))


def test_pupil_has_zero_rms(p):
    assert(p.rms == pytest.approx(0))
