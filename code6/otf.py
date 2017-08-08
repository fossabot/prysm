'''
A base optical transfer function interface
'''
import numpy as np
from numpy import floor
from numpy.fft import fft2, fftshift

from scipy import interpolate

from matplotlib import pyplot as plt

from code6.conf import config
from code6.psf import PSF
from code6.fttools import forward_ft_unit
from code6.util import correct_gamma, share_fig_ax, guarantee_array
from code6.coordinates import polar_to_cart

class MTF(object):
    '''Modulation Transfer Function

    Properties:
        tan: slice along the X axis of the MTF object.

        sag: slice along the Y axis of the MTF object.

    Instance Methods:
        exact_polar: returns the exact MTF at a given set of frequency, azimuth
            pairs.  A list of frequencies can be given to evaluate along the X
            axis only.

        exact_xy: returns the exact MTF at a given X,Y frequency pair.  Returns
            a list of MTF values.

        plot2d: Makes a 2D plot of the MTF.  Returns (fig, ax)

        plot_tan_sag: Makes a plot of the tan/sag (x/y) MTF.  Returns (fig, ax)

    Private Instance Methods:
        _make_interp_function: generates an interpolation function for the MTF,
            and stores it in the class instance.

    Static Methods:
        from_psf: Generates an MTF object from a PSF.

        from_pupil: Generates an intermediate PSF object, and MTF from that PSF.

    '''
    def __init__(self, data, unit):
        '''Creates an MTF object

        Args:
            data (:class:`numpy.ndarray`): MTF values on 2D grid
            samples (`int`): number of samples along each axis of the MTF.
                for a 256x256 MTF, samples=256
            sample_spacing (`float`): center-to-center spacing of samples,
                in frequency units.

        Returns:
            MTF:  A new :class:`MTF` instance.

        '''
        self.data = data
        self.unit = unit
        self.samples = len(unit)
        self.center = int(floor(self.samples/2))

    # quick-access slices ------------------------------------------------------

    @property
    def tan(self):
        '''Retrieves the tangential MTF
        '''
        return self.unit[self.center:-1], self.data[self.center, self.center:-1]

    @property
    def sag(self):
        '''Retrieves the sagittal MTF
        '''
        return self.unit[self.center:-1], self.data[self.center:-1, self.center]

    def exact_polar(self, freqs, azimuths=None):
        '''Retrieves the MTF at the specified frequency-azimuth pairs
        
        Args:
            freqs (iterable): radial frequencies to retrieve MTF for
            azimuths (iterable): corresponding azimuths to retrieve MTF for

        Returns:
            list: MTF at the given points

        '''
        self._make_interp_function()

        # handle user-unspecified azimuth
        if azimuths is None:
            if type(freqs) in (int, float):
                # single azimuth
                azimuths = 0
            else:
                azimuths = [0] * len(freqs)

        # handle single value case
        if type(freqs) in (int, float):
            x, y = polar_to_cart(freqs, azimuths)
            return float(self.interpf(x, y))

        outs = []
        for freq, az in zip(freqs, azimuths):
            x, y = polar_to_cart(freq, az)
            outs.append(float(self.interpf(x, y)))
        return outs

    def exact_xy(self, x, y=None):
        '''Retrieves the MTF at the specified X-Y frequency pairs

        Args:
            x (iterable): X frequencies to retrieve the MTF at
            y (iterable): Y frequencies to retrieve the MTF at

        Returns:
            list: MTF at the given points

        '''
        self._make_interp_function()

        # handle user-unspecified azimuth
        if y is None:
            if type(x) in (int, float):
                # single azimuth
                y = 0
            else:
                y = [0] * len(x)

        # handle single value case
        if type(x) in (int, float):
            return float(self.interpf(x, y))

        outs = []
        for x, y in zip(x, y):
            outs.append(float(self.interpf(x, y)))
        return outs
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
               xlim=(-max_freq, max_freq),
               ylim=(-max_freq, max_freq))
        return fig, ax

    def plot_tan_sag(self, max_freq=200, fig=None, ax=None, labels=('Tangential', 'Sagittal')):
        u, tan = self.tan
        _, sag = self.sag

        fig, ax = share_fig_ax(fig, ax)
        ax.plot(u, tan, label=labels[0], linestyle='-', lw=3)
        ax.plot(u, sag, label=labels[1], linestyle='--', lw=3)
        ax.set(xlabel='Spatial Frequency [cy/mm]',
               ylabel='MTF [Rel 1.0]',
               xlim=(0, max_freq),
               ylim=(0, 1))
        plt.legend(loc='lower left')
        return fig, ax

    # plotting -----------------------------------------------------------------

    # helpers ------------------------------------------------------------------

    def _make_interp_function(self):
        if not hasattr(self, 'interpf'):
            self.interpf = interpolate.RegularGridInterpolator((self.unit, self.unit), self.data)

        return self

    @staticmethod
    def from_psf(psf):
        dat = np.absolute(fftshift(fft2(psf.data)), dtype=config.precision)
        unit = forward_ft_unit(psf.sample_spacing, psf.samples)
        return MTF(dat / dat[psf.center, psf.center], unit)

    @staticmethod
    def from_pupil(pupil, efl, padding=1):
        psf = PSF.from_pupil(pupil, efl=efl, padding=padding)
        return MTF.from_psf(psf)

def diffraction_limited_mtf(fno=1, wavelength=0.5, num_pts=128):
    '''
    Gives the diffraction limited MTF for a circular pupil and the given parameters.
    f/# is unitless, wavelength is in microns, num_pts is length of the output array
    '''
    normalized_frequency = np.linspace(0, 1, num_pts)
    extinction = 1/(wavelength/1000*fno)
    mtf = (2/np.pi)*(np.arccos(normalized_frequency) - normalized_frequency * np.sqrt(1 - normalized_frequency**2))
    return normalized_frequency*extinction, mtf
