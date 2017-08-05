'''
Basic detector interface
'''
import numpy as np
from code6.psf import PSF
from code6.util import is_odd

class Detector(object):
    def __init__(self, pixel_size, resolution=(1024,1024), nbits=14):
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.bit_depth = nbits

    def sample_psf(self, psf):
        '''Samples a PSF, mimics capturing a photo of an oversampled representation of an image

        Args:
            PSF (code6.PSF): a point spread function

        Returns:
            PSF.  A new PSF object, as it would be sampled by the detector

        Notes:
            inspired by
            https://stackoverflow.com/questions/14916545/numpy-rebinning-a-2d-array
        '''

        # we assume the pixels are bigger than the samples in the PSF
        samples_per_pixel = int(np.ceil(self.pixel_size / psf.sample_spacing))

        # determine amount we need to trim the psf
        psf_width = 2 * psf.unit[-1]
        total_samples = int(np.floor(psf.samples / samples_per_pixel))
        output_extent = total_samples * self.pixel_size
        final_idx = total_samples * samples_per_pixel

        residual = int(psf.samples - final_idx)
        if not is_odd(residual):
            samples_to_trim = int(residual / 2)
            trimmed_data = psf.data[samples_to_trim:final_idx+samples_to_trim,
                                    samples_to_trim:final_idx+samples_to_trim]
        else:
            samples_tmp = float(residual) / 2
            samples_left = int(np.ceil(samples_tmp))
            samples_right = int(np.floor(samples_tmp))
            trimmed_data = psf.data[samples_left:final_idx+samples_right,
                                    samples_left:final_idx+samples_right]

        intermediate_view = trimmed_data.reshape(total_samples, samples_per_pixel,
                                                 total_samples, samples_per_pixel)

        output_data = intermediate_view.mean(axis=(1,3))
        return PSF(data=output_data, samples=total_samples, sample_spacing=self.pixel_size)


class OLPF(PSF):
    '''Optical Low Pass Filter.
    applies blur to an image to suppress high frequency MTF and aliasing
    '''
    def __init__(self, width_x, width_y=None, sample_spacing=0.1, samples=384):
        '''...

        Args:
            width_x (float): blur width in the x direction, expressed in microns
            width_y (float): blur width in the y direction, expressed in microns
            samples (int): number of samples in the image plane to evaluate with

        Returns:
            OLPF.  an OLPF object.
        '''

        # compute relevant spacings
        if width_y is None:
            width_y = width_x

        self.width_x = width_x
        self.width_y = width_y

        space_x = width_x / 2
        space_y = width_y / 2
        shift_x = int(np.floor(space_x / sample_spacing))
        shift_y = int(np.floor(space_y / sample_spacing))
        center  = int(np.floor(samples/2))

        data = np.zeros((samples, samples))

        data[center-shift_x, center-shift_y] = 1
        data[center-shift_x, center+shift_y] = 1
        data[center+shift_x, center-shift_y] = 1
        data[center+shift_x, center+shift_y] = 1
        super().__init__(data=data, samples=samples, sample_spacing=sample_spacing)

    def analytic_ft(self, unit_x, unit_y):
        '''Analytic fourier transform of a pixel aperture
            
            Args:
                unit_x (numpy.ndarray): sample points in x axis
                unit_y (numpy.ndarray): sample points in y axis

            Returns:
                numpy.ndarray.  2D numpy array containing the analytic fourier transform
        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (np.cos(2*xq*self.width_x/1e3)*np.cos(2*yq*self.width_y/1e3)).astype(np.complex128)

class PixelAperture(PSF):
    '''creates an image plane view of the pixel aperture
    '''
    def __init__(self, size, sample_spacing=0.1, samples=384):
        self.size = size

        center = int(np.floor(samples/2))
        half_width = size / 2
        steps = int(np.floor(half_width / sample_spacing))
        pixel_aperture = np.zeros((samples, samples))
        pixel_aperture[center-steps:center+steps, center-steps:center+steps] = 1
        super().__init__(data=pixel_aperture, sample_spacing=sample_spacing, samples=samples)

    def analytic_ft(self, unit_x, unit_y):
        '''Analytic fourier transform of a pixel aperture
        
        Args:
            unit_x (numpy.ndarray): sample points in x axis
            unit_y (numpy.ndarray): sample points in y axis

        Returns:
            numpy.ndarray.  2D numpy array containing the analytic fourier transform
        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (np.sinc(xq*self.size/1e3)*np.sinc(yq*self.size/1e3)).astype(np.complex128)

def generate_mtf(pixel_pitch=1, azimuth=0, num_samples=128):
    '''
    generates the diffraction-limited MTF for a given pixel size and azimuth w.r.t. the pixel grid
    pixel pitch in units of microns, azimuth in units of degrees
    '''
    pitch_unit = pixel_pitch / 1000
    normalized_frequencies = np.linspace(0, 2, num_samples)
    otf = np.sinc(normalized_frequencies)
    mtf = np.abs(otf)
    return normalized_frequencies/pitch_unit, mtf
