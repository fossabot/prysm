''' Detector-related simulations
'''
import numpy as np

from scipy.misc import imsave

from prysm.conf import config
from prysm.psf import PSF
from prysm.objects import Image
from prysm.util import is_odd, share_fig_ax

class Detector(object):
    def __init__(self, pixel_size, resolution=(1024,1024), nbits=14):
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.bit_depth = nbits
        self.captures = []

    def sample_psf(self, psf):
        '''Samples a PSF, mimics capturing a photo of an oversampled representation of an image

        Args:
            PSF (prysm.PSF): a point spread function

        Returns:
            PSF.  A new PSF object, as it would be sampled by the detector

        Notes:
            inspired by https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array

        '''

        # we assume the pixels are bigger than the samples in the PSF
        samples_per_pixel = int(np.ceil(self.pixel_size / psf.sample_spacing))

        # determine amount we need to trim the psf
        total_samples_x = psf.samples_x // samples_per_pixel
        total_samples_y = psf.samples_y // samples_per_pixel
        final_idx_x = total_samples_x * samples_per_pixel
        final_idx_y = total_samples_y * samples_per_pixel

        residual_x = int(psf.samples_x - final_idx_x)
        residual_y = int(psf.samples_y - final_idx_y)
        if not is_odd(residual_x):
            samples_to_trim_x = residual_x // 2
            samples_to_trim_y = residual_y // 2
            trimmed_data = psf.data[samples_to_trim_x:final_idx_x+samples_to_trim_x,
                                    samples_to_trim_y:final_idx_y+samples_to_trim_y]
        else:
            samples_tmp_x = float(residual_x) / 2
            samples_tmp_y = float(residual_y) / 2
            samples_top = int(np.ceil(samples_tmp_y))
            samples_bottom = int(np.ceil(samples_tmp_y))
            samples_left = int(np.ceil(samples_tmp_x))
            samples_right = int(np.floor(samples_tmp_x))
            trimmed_data = psf.data[samples_left:final_idx_x+samples_right,
                                    samples_bottom:final_idx_y+samples_top]

        intermediate_view = trimmed_data.reshape(total_samples_x, samples_per_pixel,
                                                 total_samples_y, samples_per_pixel)

        output_data = intermediate_view.mean(axis=(1, 3))

        self.captures.append(PSF(data=output_data,
                                 sample_spacing=self.pixel_size))
        return self.captures[-1]

    def sample_image(self, image):
        ''' Samples an image.

        Args:
            image (`Image`): an Image object.

        Returns:
            `Image`: a new, sampled image.

        '''
        intermediate_psf = self.sample_psf(image.as_psf())
        self.captures.append(Image(data=intermediate_psf.data,
                                   sample_spacing=intermediate_psf.sample_spacing))
        return self.captures[-1]

    def save_image(self, path, which='last'):
        ''' Saves an image captured by the detector

        Args:
            path (`string`): path to save the image to

            which (`string` or `int`): if string, "first" or "last", otherwise
                index into the capture buffer of the camera.

        Returns:
            null: no return.

        '''
        if which.lower() == 'last':
            self.captures[-1].save(path, self.bit_depth)
        elif type(which) is int:
            self.captures[which].save(path, self.bit_depth)
        else:
            raise ValueError('invalid "which" provided')

    def show_image(self, which='last', fig=None, ax=None):
        ''' Shows an image captured by the detector

        Args:
            which (`string` or `int`): if string, "first" or "last", otherwise
                index into the capture buffer of the camera.

            fig (`matplotlib.figure`): Figure to display in.

            ax (`maxplotlib.axis`): Axis to display in.

        Returns:
            `tuple` containing:

                `matplotlib.figure`: Figure containing the image.

                `matplotlib.axis`: Axis contianing the image.

        '''

        if which.lower() == 'last':
            fig, ax = self.captures[-1].show(fig=fig, ax=ax)
        elif type(which) is int:
            fig, ax = self.captures[which].show(fig=fig, ax=ax)
        return fig, ax

class OLPF(PSF):
    ''' Optical Low Pass Filter.
        applies blur to an image to suppress high frequency MTF and aliasing
        when combined with a PixelAperture.
    '''
    def __init__(self, width_x, width_y=None, sample_spacing=0.1, samples_x=384, samples_y=None):
        ''' Creates a new OLPF object.

        Args:
            width_x (`float`): blur width in the x direction, expressed in microns.

            width_y (`float`): blur width in the y direction, expressed in microns.

            samples (int): number of samples in the image plane to evaluate with

        Returns:
            `OLPF`: an OLPF object.

        '''

        # compute relevant spacings
        if width_y is None:
            width_y = width_x
        if samples_y is None:
            samples_y = samples_x

        self.width_x = width_x
        self.width_y = width_y

        space_x = width_x / 2
        space_y = width_y / 2
        shift_x = int(space_x // sample_spacing)
        shift_y = int(space_y // sample_spacing)
        center_x  = samples_x // 2
        center_y = samples_y // 2

        data = np.zeros((samples_x, samples_y))

        data[center_y-shift_y, center_x-shift_x] = 1
        data[center_y-shift_y, center_x+shift_x] = 1
        data[center_y+shift_y, center_x-shift_x] = 1
        data[center_y+shift_y, center_x+shift_x] = 1
        super().__init__(data=data, sample_spacing=sample_spacing)

    def analytic_ft(self, unit_x, unit_y):
        ''' Analytic fourier transform of a pixel aperture

        Args:
            unit_x (numpy.ndarray): sample points in x axis.

            unit_y (numpy.ndarray): sample points in y axis.

        Returns:
            `numpy.ndarray`: 2D numpy array containing the analytic fourier transform.

        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (np.cos(2 * xq * self.width_x / 1e3) * \
                np.cos(2 * yq * self.width_y / 1e3)).astype(config.precision)

class PixelAperture(PSF):
    ''' The aperture of a pixel.
    '''
    def __init__(self, size_x, size_y=None, sample_spacing=0.1, samples_x=384, samples_y=None):
        ''' Creates a new PixelAperture object.

        Args:
            size_x (`float`): size of the aperture in the x dimension, in microns.

            size_y (`float`): siez of the aperture in teh y dimension, in microns.

            sample_spacing (`float`): spacing of samples, in microns.

            samples_x (`int`): number of samples in the x dimension.

            samples_y (`int`): number of samples in the y dimension.

        Returns:
            `PixelAperture`: a new PixelAperture object.

        '''
        if size_y is None:
            size_y = size_x
        if samples_y is None:
            samples_y = samples_x

        self.size_x = size_x
        self.size_y = size_y
        self.center_x = samples_x // 2
        self.center_y = samples_y // 2

        half_width = size_x / 2
        half_height = size_y / 2
        steps_x = int(half_width // sample_spacing)
        steps_y = int(half_height // sample_spacing)

        pixel_aperture = np.zeros((samples_x, samples_y))
        pixel_aperture[self.center_y-steps_y:self.center_y+steps_y,
                       self.center_x-steps_x:self.center_x+steps_x] = 1
        super().__init__(data=pixel_aperture, sample_spacing=sample_spacing)

    def analytic_ft(self, unit_x, unit_y):
        ''' Analytic fourier transform of a pixel aperture.

        Args:
            unit_x (`numpy.ndarray`): sample points in x axis.

            unit_y (`numpy.ndarray`): sample points in y axis.

        Returns:
            `numpy.ndarray`: 2D numpy array containing the analytic fourier transform.

        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (np.sinc(xq*self.size_x/1e3) * \
                np.sinc(yq*self.size_y/1e3)).astype(config.precision)

def generate_mtf(pixel_aperture=1, azimuth=0, num_samples=128):
    ''' generates the 1D diffraction-limited MTF for a given pixel size and azimuth.

    Args:
        pixel_aperture (`float`): aperture of the pixel, in microns.  Pixel is
            assumed to be square.

        azimuth (`float`): azimuth to retrieve the MTF at, in degrees.

        num_samples (`int`): number of samples in the output array.

    Returns:
        `tuple` containing:
            `numpy.ndarray`: array of units, in cy/mm.

            `numpy.ndarray`: array of MTF values (rel. 1.0).

    Notes:
        Azimuth is not actually implemented yet.
    '''
    pitch_unit = pixel_aperture / 1e3
    normalized_frequencies = np.linspace(0, 2, num_samples)
    otf = np.sinc(normalized_frequencies)
    mtf = np.abs(otf)
    return normalized_frequencies/pitch_unit, mtf
