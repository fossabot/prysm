'''
A base point spread function interface
'''
import numpy as np
from numpy import floor
from numpy import power as npow
from numpy.fft import fft2, fftshift, ifftshift, ifft2

from matplotlib import pyplot as plt

from code6.fttools import pad2d, forward_ft_unit
from code6.coordinates import cart_to_polar, polar_to_cart, uniform_cart_to_polar, resample_2d_complex
from code6.util import pupil_sample_to_psf_sample, correct_gamma, share_fig_ax, fold_array

class PSF(object):
    '''Point Spread Function representations

    Notes:
        Subclasses must implement an analyic_ft method with signature a_ft(unit_x, unit_y)
    '''
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
        return self.unit, self.data[self.center, :]

    @property
    def slice_y(self):
        '''
        Retrieves a slices through the y axis of the PSF
        '''
        return self.unit, self.data[:, self.center]


    def encircled_energy(self, azimuth=None):
        '''
        returns the encircled energy at the requested azumith.  If azimuth is None, returns the
        azimuthal average
        '''

        rho, phi, interp_dat = uniform_cart_to_polar(self.unit, self.unit, self.data)
        avg_fold = fold_array(interp_dat)

        if azimuth is None:
            # take average of all azimuths as input data
            dat = np.average(avg_fold, axis=0)
        else:
            index = np.searchsorted(phi, np.radians(azimuth))
            dat = avg_fold[index, :]

        enc_eng = np.cumsum(dat)
        enc_eng /= enc_eng[-1]
        return self.unit[self.center:], enc_eng

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, log=False, axlim=25, interp_method='bicubic', fig=None, ax=None):
        if log:
            fcn = 20 * np.log10(1e-100 + self.data)
            label_str = 'Normalized Intensity [dB]'
            lims = (-100, 0) # show first 100dB -- range from (1e-6, 1) in linear scale
        else:
            fcn = correct_gamma(self.data)
            label_str = 'Normalized Intensity [a.u.]'
            lims = (0, 1)

        left, right = self.unit[0], self.unit[-1]

        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(fcn,
                       extent=[left, right, left, right],
                       cmap='Greys_r',
                       interpolation=interp_method,
                       clim=lims)
        fig.colorbar(im, label=label_str, ax=ax, fraction=0.046)
        ax.set(xlabel=r'Image Plane X [$\mu m$]',
               ylabel=r'Image Plane Y [$\mu m$]',
               xlim=(-axlim, axlim),
               ylim=(-axlim, axlim))
        return fig, ax

    def plot_slice_xy(self, log=False, axlim=20):
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
               xlim=(-axlim, axlim),
               ylim=lims)
        plt.legend(loc='upper right')
        return fig, ax

    def plot_encircled_energy(self):
        unit, data = self.encircled_energy()

        fig, ax = plt.subplots()
        ax.plot(unit, data, lw=3)
        ax.set(xlabel=r'Image Plane Distance [$\mu m$]',
               ylabel=r'Encircled Energy [Rel 1.0]',
               xlim=(0, 20))
        return fig, ax

    # plotting -----------------------------------------------------------------

    # helpers ------------------------------------------------------------------

    def conv(self, psf2):
        if issubclass(psf2.__class__, PSF):
            # subclasses have analytic fourier transforms and we can exploit this for high speed,
            # aliasing-free convolution
            psf_ft = fft2(self.data)
            psf_unit = forward_ft_unit(self.sample_spacing, self.samples)
            psf2_ft = fftshift(psf2.analytic_ft(psf_unit, psf_unit))
            psf3 = PSF(data=np.absolute(ifft2(psf_ft * psf2_ft)),
                       samples=self.samples,
                       sample_spacing=self.sample_spacing)
            return psf3._renorm()
        return convpsf(self, psf2)

    def _renorm(self):
        self.data /= self.data.max()
        return self

    # helpers ------------------------------------------------------------------

    @staticmethod
    def from_pupil(pupil, efl, padding=1):
        '''
        Uses fresnel diffraction to propogate a pupil and compute a point spread function
        '''
        # padded pupil contains 1 pupil width on each side for a width of 3
        psf_samples = (pupil.samples * padding) * 2 + pupil.samples
        sample_spacing = pupil_sample_to_psf_sample(pupil_sample=pupil.sample_spacing * 1000,
                                                    num_samples=psf_samples,
                                                    wavelength=pupil.wavelength,
                                                    efl=efl)
        padded_wavefront = pad2d(pupil.fcn, padding)
        impulse_response = ifftshift(fft2(fftshift(padded_wavefront)))
        psf = npow(abs(impulse_response), 2)
        return PSF(psf / np.max(psf), psf_samples, sample_spacing)


def convpsf(psf1, psf2):
    if psf2.samples == psf1.samples and psf2.sample_spacing == psf1.sample_spacing:
        # no need to interpolate, use FFTs to convolve
        psf3 = PSF(data=np.absolute(ifftshift(ifft2(fft2(psf1.data) * fft2(psf2.data)))),
                   samples=psf1.samples,
                   sample_spacing=psf1.sample_spacing)
        return psf3._renorm()
    else:
        # need to interpolate, suppress all frequency content above nyquist for the less sampled psf
        if psf1.sample_spacing > psf2.sample_spacing:
            # psf1 has the lower nyquist, resample psf2 in the fourier domain to match psf1
            return _unequal_spacing_conv_core(psf1, psf2)
        else:
            # psf2 has lower nyquist, resample psf1 in the fourier domain to match psf2
            return _unequal_spacing_conv_core(psf2, psf1)

def _unequal_spacing_conv_core(psf1, psf2):
    ft1 = fft2(psf1.data)
    unit1 = forward_ft_unit(psf1.sample_spacing, psf1.samples)
    ft2 = fft2(psf2.data)
    unit2 = forward_ft_unit(psf2.sample_spacing, psf2.samples)
    ft3 = resample_2d_complex(ft2, (unit2, unit2), (unit1, unit1[::-1]))
    psf3 = PSF(data=np.absolute(ifftshift(ifft2(ft1 * ft3))),
               samples=psf1.samples,
               sample_spacing=psf2.sample_spacing)
    return psf3._renorm()
