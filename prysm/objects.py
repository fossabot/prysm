''' Object to convolve lens PSFs with
'''
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from scipy.misc import imsave

from matplotlib import pyplot as plt

from prysm.util import correct_gamma
from prysm.fttools import forward_ft_unit
from prysm.psf import PSF, _unequal_spacing_conv_core

class Image(object):
    ''' Images of an object
    '''
    def __init__(self, data, sample_spacing):
        ''' Creates a new Image object.

        Args:
            data (`numpy.ndarray`): data that represents the image, 2D.

            sample-spacing (`float`): pixel pitch of the data.

        Returns:
            `Image`: new Image object.

        '''
        self.data = data
        self.sample_spacing = sample_spacing
        self.samples_x, self.samples_y = data.shape
        self.center_x, self.center_y = self.samples_x // 2, self.samples_y // 2
        self.ext_x = sample_spacing * self.center_x
        self.ext_y = sample_spacing * self.center_y

    def show(self, interp_method=None):
        ''' Displays the image.
        '''
        ex, ey = self.ext_x, self.ext_y
        lims = (0,1)
        fig, ax = plt.subplots()
        ax.imshow(self.data,#correct_gamma(self.data),
                  #extent=[-ex, ex, -ey, ey],
                  cmap='Greys_r',
                  interpolation=interp_method,
                  clim=lims,
                  origin='lower')
        return fig, ax

    def as_psf(self):
        ''' Converts this image to a PSF object.
        '''
        return PSF(self.data, self.samples_x, self.sample_spacing)

    def convpsf(self, psf):
        ''' Convolves with a PSF for image simulation

        Args:
            psf (`PSF`): a PSF

        Returns:
            `Image`: A new, blurred image.

        '''
        img_psf = self.as_psf()
        conved_image = _unequal_spacing_conv_core(img_psf, psf)
        #return conved_image
        return Image(data=conved_image.data,
                     sample_spacing=self.sample_spacing)

    def save(self, path, nbits=8):
        ''' Write the image to a png, jpg, tiff, etc.

        Args:
            path (`string`): path to write the image to.

            nbits (`int`): number of bits in the output image.

        Returns:
            null: no return

        '''
        dat = (self.data * 255).astype(np.uint8)
        imsave(path, dat)


class Slit(Image):
    ''' Representation of a single slit.
    '''
    def __init__(self, width, orientation='Vertical', sample_spacing=0.075, samples=384):
        ''' Creates a new Slit instance.

        Args:
            width (`float`): the width of the slit.

            orientation (`string`): the orientation of the slit, Horizontal, Vertical, or Crossed / Both

            sample_spacing (`float`): spacing of samples in the synthetic image.

            samples (`int`): number of samples per dimension in the synthetic image.

        Returns:
            `numpy.ndarray`: array containing the synthetic slit.

        '''

        # produce coordinate arrays
        ext = samples / 2 * sample_spacing
        x, y = np.linspace(-ext, ext, samples), np.linspace(-ext, ext, samples)
        w = width / 2

        # produce the background
        arr = np.zeros((samples, samples))

        # paint in the slit
        if orientation.lower() in ('v', 'vert', 'vertical'):
            arr[:, abs(x)<w] = 1
        elif orientation.lower() in ('h', 'horiz', 'horizontal'):
            arr[abs(y)<w, :] = 1
        elif orientation.lower() in ('b', 'both', 'c', 'crossed'):
            arr[abs(y)<w, :] = 1
            arr[:, abs(x)<w] = 1

        super().__init__(data=arr, sample_spacing=sample_spacing)

class Pinhole(Image):
    ''' Representation of a pinhole object.
    '''
    def __init__(self, width, sample_spacing=0.025, samples=384):
        ''' Produces a Pinhole.

        Args:
            width (`float`): the width of the pinhole.

            sample_spacing (`float`): spacing of samples in the synthetic image.

            samples (`int`): number of samples per dimension in the synthetic image.

        Returns:
            `numpy.ndarray`: array containing the synthetic pinhole.

        '''

        # produce coordinate arrays
        ext = samples / 2 * sample_spacing
        x, y = np.linspace(-ext, ext, samples), np.linspace(-ext, ext, samples)
        xv, yv = np.meshgrid(x,y)
        w = width / 2

        # produce the background
        arr = np.zeros((samples, samples))
        arr[np.sqrt(xv**2 + yv**2) < w] = 1
        super().__init__(data=arr, sample_spacing=sample_spacing)