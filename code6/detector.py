'''
Basic detector interface
'''
import numpy as np
from code6.psf import PSF

px_size_default = 5                    # um
resolution_default = (1024, 1024)      # px * px
noise_default = dict(read=5,           # e-
    bias=200,                          # DN
    dark=0.2,                          # e-/s
    prnu=np.zeros(resolution_default), # e- per pixel
    dsnu=np.zeros(resolution_default)  # e- per pixel
)
wavelengths_default = np.arange(400, 700, 10)    # nm
qe_default = (wavelengths_default-700)/700 + 0.5 # e-/photon
light_default = dict(wavelengths=wavelengths_default,
    qe=qe_default,
    fwc=50000 # e-
)

class Detector(object):
    def __init__(self, pixel_size=px_size_default, resolution=resolution_default, noise=noise_default, light=light_default):
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.noise = noise
        self.light = light

    def expose(self, truth_img, truth_wavelengths, ts=1/100):
        # start from nothing
        output_image = np.zeros((self.resolution, self.resolution))

        # add the bias
        output_image += self.noise['bias']

        # add the dark current
        output_image += (np.random.random(self.resolution, self.resolution) * self.noise['dark']*ts) + self.noise['dsnu']

        # add the read noise
        output_image += np.random.random((self.resolution, self.resolution)) * self.noise['read']

        # add the signal
        spectral_content = np.dot(self.light['qe'], truth_wavelengths[:,:,2])
        output_image += truth_img * spectral_content * ts
        
        return output_image


class ADC(object):
    def __init(self, precision=16, noise=5):
        self.precision = precision
        self.noise = noise


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
        space_x = width_x / 2
        space_y = width_y / 2
        shift_x = int(np.floor(space_x / sample_spacing))
        shift_y = int(np.floor(space_y / sample_spacing))
        center  = int(np.floor(samples/2))
        
        data = np.zeros((samples, samples))

        data[center-shift_x, center-shift_y] = 1##0.75
        data[center-shift_x, center+shift_y] = 1#0.75
        data[center+shift_x, center-shift_y] = 1#0.75
        data[center+shift_x, center+shift_y] = 1#0.75
        super().__init__(data=data, samples=samples, sample_spacing=sample_spacing)

class PixelAperture(PSF):
    '''creates a PSF view of the pixel aperture
    '''
    def __init__(self, size, sample_spacing=0.1, samples=384):
        center = int(np.floor(samples/2))
        half_width = size / 2
        steps = int(np.floor(half_width / sample_spacing))
        pixel_aperture = np.zeros((samples, samples))
        pixel_aperture[center-steps:center+steps, center-steps:center+steps] = 1
        super().__init__(data=pixel_aperture, sample_spacing=sample_spacing, samples=samples)

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