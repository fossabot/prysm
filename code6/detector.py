'''
Basic detector interface
'''
import numpy as np

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

def generate_mtf(pixel_pitch=1, azimuth=0, num_samples=128):
    '''
    generates the diffraction-limited MTF for a given pixel size and azimuth w.r.t. the pixel grid
    pixel pitch in units of microns, azimuth in units of degrees
    '''
    pitch_unit = pixel_pitch / 1000
    normalized_frequencies = np.linspace(0, 2, num_samples)
    otf = np.sinc(2 * np.pi * normalized_frequencies / pixel_pitch)
    mtf = np.abs(otf)
    return normalized_frequencies/pitch_unit, mtf