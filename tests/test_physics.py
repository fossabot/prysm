''' Unit tests for the physics of prysm.
'''
import numpy as np

from prysm import Pupil, PSF
from prysm.psf import airydisk

PRECISION = 1e-3  # ~0.1%


def test_diffprop_matches_airydisk():
    epd = 1
    efl = 10
    fno = efl / epd
    wavelength = 0.5

    p = Pupil(wavelength=wavelength, epd=epd)
    psf = PSF.from_pupil(p, efl)
    u, sx = psf.slice_x
    u, sy = psf.slice_y
    analytic = airydisk(u, fno, wavelength)
    assert np.allclose(sx, analytic, rtol=1e-3, atol=1e-3)
    assert np.allclose(sy, analytic, rtol=1e-3, atol=1e-3)
