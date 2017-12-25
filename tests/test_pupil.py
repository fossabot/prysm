''' Unit tests for pupil objects
'''
import pytest

import matplotlib
from matplotlib import pyplot as plt

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
    assert(p._opd_unit == parameters['opd_unit'])


def test_pupil_plot2d_makes_own_fig_and_ax(p):
    return_values = p.plot2d()
    assert(type(return_values[0]) is matplotlib.figure.Figure)  # first return is a figure
    assert(len(return_values) == 2)  # figure and axis return values


def test_pupil_plot2d_shares_fig_and_ax(p):
    fig, ax = plt.subplots()
    fig2, ax2 = p.plot2d()
    assert(fig.number == fig2.number)


def test_pupil_interferogram_makes_own_fig_and_ax(p):
    return_values = p.plot2d()
    assert(type(return_values[0]) is matplotlib.figure.Figure)  # first return is a figure
    assert(len(return_values) == 2)  # figure and axis return values


def test_pupil_interferogram_shares_fig_and_ax(p):
    fig, ax = plt.subplots()
    fig2, ax2 = p.plot2d()
    assert(fig.number == fig2.number)


def test_pupil_plot_slice_xy_makes_own_fig_and_ax(p):
    return_values = p.plot2d()
    assert(type(return_values[0]) is matplotlib.figure.Figure)  # first return is a figure
    assert(len(return_values) == 2)  # figure and axis return values


def test_pupil_plot_slice_xy_shares_fig_and_ax(p):
    fig, ax = plt.subplots()
    fig2, ax2 = p.plot2d()
    assert(fig.number == fig2.number)
