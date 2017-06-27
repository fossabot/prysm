'''
Base meta-class for aberration simulations that provides wavefront, PSF, and MTF simulation methods
'''
import numpy as np
from numpy import arctan2, exp, floor, nan, sqrt, pi
from numpy import power as npow
from numpy.fft import fft2, fftshift, ifftshift
from matplotlib import pyplot as plt

from code6.lens import Lens
from code6.fringezernike import FringeZernike
from code6.simulation import Simulation
from code6.psf import PSF
from code6.otf import MTF
from code6.fttools import pad2d
from code6.util import pupil_sample_to_psf_sample, psf_sample_to_pupil_sample

class Engine(object):
    def __init__(self, parameters=Simulation(), lens=Lens(), pupil=FringeZernike()):
        epd = lens.efl / lens.fno
        rho = epd / 2
        psf_samples = (parameters.padding * parameters.samples) + parameters.samples

        self.parameters = parameters
        self.lens = lens
        self.pupil = pupil

        # and bools indicating if things are computed
        self._pupil_computed = False
        self._psf_computed = False
        self._mtf_computed = False

    # helper methods -----------------------------------------------------------

    def _check_pupil(self):
        if not self._pupil_computed:
            # self.pupil_phase, self.pupil_fcn = 
            self.pupil.build(wavelength=self.parameters.wavelength)
            self.pupil.clip()
            self._pupil_computed = True

    def _check_psf(self):
        if not self._psf_computed:
            self._check_pupil()
            self.psf = PSF.from_pupil(self.pupil,
                                      self.parameters.wavelength,
                                      self.lens.efl,
                                      self.parameters.padding)
            self._psf_computed = True

    def _check_mtf(self):
        if not self._mtf_computed:
            self._check_psf()
            self.mtf = MTF.from_psf(self.psf)
            self._mtf_computed = True

    def plot_fourier_chain(self):
        self._check_mtf()
        # generate a figure
        fig, ax = plt.subplots(nrows=1, ncols=3)

        # glue in the desired plots
        _, pupil_axis = self.pupil.plot2d()
        ax[0] = pupil_axis

        _, psf_axis = self.psf.plot2d()
        ax[1] = psf_axis

        _, mtf_axis = self.mtf.plot_tan_sag()
        ax[2] = mtf_axis
        fig.show()
        return fig, ax

    def __repr__(self):
        return 'Engine'

    # helper methods -----------------------------------------------------------
