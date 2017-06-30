'''
A base point spread function interface
'''
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
from numpy import power as npow
from numpy import floor
from matplotlib import pyplot as plt

from code6.util import pupil_sample_to_psf_sample, correct_gamma
from code6.fttools import pad2d

class PSF(object):
    def __init__(self, data, samples, sample_spacing):
        # dump inputs into class instance
        self.data = data
        self.samples = samples
        self.sample_spacing = sample_spacing
        self.center = int(floor(samples/2))

        # compute ordinate axis
        ext = self.sample_spacing * samples / 2
        self.unit = np.linspace(-ext, ext, samples)

    # quick-access slices ------------------------------------------------------

    @property
    def slice_x(self):
        '''
        Retrieves a slice through the x axis of the PSF
        '''
        return self.unit, self.data[self.center,:]

    @property
    def slice_y(self):
        '''
        Retrieves a slices through the y axis of the PSF
        '''
        return self.unit, self.data[:, self.center]

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, log=False):
        if log:
            fcn = 20 * np.log10(1e-100 + self.data)
            label_str = 'Normalized Intensity [dB]'
            lims = (-100, 0) # show first 100dB -- range from (1e-6, 1) in linear scale
        else:
            fcn = correct_gamma(self.data)
            label_str = 'Normalized Intensity [a.u.]'
            lims = (0, 1)

        left, right = self.unit[0], self.unit[-1]

        fig, ax = plt.subplots()
        im = ax.imshow(fcn,
                       extent=[left, right, left, right],
                       cmap='Greys_r',
                       interpolation='bicubic',
                       clim=lims)
        fig.colorbar(im, label=label_str)
        ax.set(xlabel=r'Image Plane X [$\mu m$]',
               ylabel=r'Image Plane Y [$\mu m$]',
               xlim=(-10,10),
               ylim=(-10,10))
        return fig, ax

    def plot_slice_xy(self, log=False):
        u, x = self.slice_x
        _, y = self.slice_y
        if log:
            fcn_x = 20 * np.log10(1e-100 + x)
            fcn_y = 20 * np.log10(1e-100 + y)
            label_str = 'Normalized Intensity [dB]'
            lims = (-120, 0)
        else:
            fcn_x = x
            fcn_y = y
            label_str = 'Normalized Intensity [a.u.]'
            lims = (0, 1)

        fig, ax = plt.subplots()
        ax.plot(u, fcn_x, label='Slice X', lw=3)
        ax.plot(u, fcn_y, label='Slice Y', lw=3)
        ax.set(xlabel=r'Image Plane X [$\mu m$]',
               ylabel=label_str,
               xlim=(-10,10),
               ylim=lims)
        plt.legend(loc='upper right')
        return fig, ax

    # plotting -----------------------------------------------------------------
    
    @staticmethod
    def from_pupil(pupil, wavelength, efl, padding=1):
        '''
        Uses fresnel diffraction to propogate a pupil and compute a point spread function
        '''
        # padded pupil contains 1 pupil width on each side for a width of 3
        psf_samples = (pupil.samples * padding) * 2 + pupil.samples
        sample_spacing = pupil_sample_to_psf_sample(pupil_sample=pupil.sample_spacing * 1000,
                                                    num_samples=psf_samples,
                                                    wavelength=wavelength,
                                                    efl=efl)
        padded_wavefront = pad2d(pupil.fcn, padding)
        impulse_response = ifftshift(fft2(fftshift(padded_wavefront)))
        psf = npow(abs(impulse_response), 2)
        return PSF(psf / np.max(psf), psf_samples, sample_spacing)
