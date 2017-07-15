'''
A base optical transfer function interface
'''
import numpy as np
from numpy import floor
from numpy.fft import fft2, fftshift, ifftshift

from matplotlib import pyplot as plt

from code6.psf import PSF
from code6.util import correct_gamma, share_fig_ax

class MTF(object):
    def __init__(self, data, unit):
        # dump inputs into class instance
        self.data = data
        self.unit = unit
        self.samples = len(unit)
        self.center = int(floor(self.samples/2))

    # quick-access slices ------------------------------------------------------

    @property
    def tan(self):
        '''
        Retrieves the tangential MTF
        '''
        return self.unit[self.center:-1], self.data[self.center, self.center:-1]

    @property
    def sag(self):
        '''
        Retrieves the sagittal MTF
        '''
        return self.unit[self.center:-1], self.data[self.center:-1, self.center]

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, log=False, max_freq=200, fig=None, ax=None):
        if log:
            fcn = 20 * np.log10(1e-24 + self.data)
            label_str = 'MTF [dB]'
            lims = (-120, 0)
        else:
            fcn = correct_gamma(self.data)
            label_str = 'MTF [Rel 1.0]'
            lims = (0, 1)

        left, right = self.unit[0], self.unit[-1]

        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(fcn,
                       extent=[left, right, left, right],
                       cmap='Greys_r',
                       interpolation='bicubic',
                       clim=lims)
        fig.colorbar(im, label=label_str, ax=ax, fraction=0.046)
        ax.set(xlabel='Spatial Frequency X [cy/mm]',
               ylabel='Spatial Frequency Y [cy/mm]',
               xlim=(-max_freq,max_freq),
               ylim=(-max_freq,max_freq))
        return fig, ax

    def plot_tan_sag(self, max_freq=200):
        u, tan = self.tan
        _, sag = self.sag

        fig, ax = plt.subplots()
        ax.plot(u, tan, label='Tangential', linestyle='-', lw=3)
        ax.plot(u, sag, label='Sagittal', linestyle='--', lw=3)
        ax.set(xlabel='Spatial Frequency [cy/mm]',
               ylabel='MTF [Rel 1.0]',
               xlim=(0,max_freq),
               ylim=(0,1))
        plt.legend(loc='lower left')
        return fig, ax

    # plotting -----------------------------------------------------------------

    @staticmethod
    def from_psf(psf):
        dat = abs(fftshift(fft2(psf.data)))
        f_s = int(floor(psf.samples / 2))
        unit = 1 / (psf.sample_spacing / 1e3) * range(-f_s, f_s) / psf.samples
        return MTF(dat/dat[f_s,f_s], unit)

    @staticmethod
    def from_pupil(pupil, efl):
        psf = PSF.from_pupil(pupil, efl=efl)
        return __class__.from_psf(psf)

def diffraction_limited_mtf(fno=1, wavelength=0.5, num_pts=128):
    '''
    Gives the diffraction limited MTF for a circular pupil and the given parameters.
    f/# is unitless, wavelength is in microns, num_pts is length of the output array
    '''
    normalized_frequency = np.linspace(0, 1, num_pts)
    extinction = 1/(wavelength/1000*fno)
    mtf = (2/np.pi)*(np.arccos(normalized_frequency) - normalized_frequency * np.sqrt(1 - normalized_frequency**2))
    return normalized_frequency*extinction, mtf