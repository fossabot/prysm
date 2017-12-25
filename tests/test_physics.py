''' Unit tests for the physics of prysm.
'''
import numpy as np

import pytest

from prysm import Pupil, PSF, MTF
from prysm.psf import airydisk
from prysm.otf import diffraction_limited_mtf

PRECISION = 1e-3  # ~0.1%

TEST_PARAMETERS = [
    (10.0, 1.000, 0.5),  # f/10, visible light
    (10.0, 1.000, 1.0),  # f/10, SWIR light
    (3.00, 1.125, 3.0)]  # f/2.66666, MWIR light


@pytest.mark.parametrize('efl, epd, wvl', TEST_PARAMETERS)
def test_diffprop_matches_airydisk(efl, epd, wvl):
    fno = efl / epd

    p = Pupil(wavelength=wvl, epd=epd)
    psf = PSF.from_pupil(p, efl)
    u, sx = psf.slice_x
    u, sy = psf.slice_y
    analytic = airydisk(u, fno, wvl)
    assert np.allclose(sx, analytic, rtol=PRECISION, atol=PRECISION)
    assert np.allclose(sy, analytic, rtol=PRECISION, atol=PRECISION)


@pytest.mark.parametrize('efl, epd, wvl', TEST_PARAMETERS)
def test_diffprop_matches_analyticmtf(efl, epd, wvl):
    fno = efl / epd
    p = Pupil(wavelength=wvl, epd=epd)
    psf = PSF.from_pupil(p, efl)
    mtf = MTF.from_psf(psf)
    u, t = mtf.tan
    uu, s = mtf.sag

    analytic_1 = diffraction_limited_mtf(fno, wvl, frequencies=u)
    analytic_2 = diffraction_limited_mtf(fno, wvl, frequencies=uu)
    assert np.allclose(analytic_1, t, rtol=PRECISION, atol=PRECISION)
    assert np.allclose(analytic_2, s, rtol=PRECISION, atol=PRECISION)
