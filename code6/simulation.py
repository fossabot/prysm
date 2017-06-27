'''
A container class for simulation parameters.
'''
import warnings
_image_planes = {
        'psf',
        'point spread function',
        'img',
        'image',
        'image plane'
}
_pupil_planes = {
    'pupil',
    'exit pupil',
}
_valid_sample_planes = _image_planes | _pupil_planes

_opd_units = [
    'waves',
    'lambda',
    '$\lambda$',
    'nanometers',
    'nm',
    'microns',
    'micrometers',
    'um',
    'μm',
    '$\mu m$'
]

class Simulation(object):
    '''
    Describes the parameters used to perform fourier optics simulation
    '''
    def __init__(self, wavelength=0.5, samples=256, padding=1, sample_plane='pupil', sample_size=0.25, opd_unit='$\lambda$'):
        # wavelength  in um
        # sample_size in um
        if wavelength < 0:
            raise ValueError('Wavelength must be positive')
        if samples > 2048:
            warnings.warn('Dense sampling may result in out of memory errors', UserWarning)
        if padding < 1:
            raise UserWarning('Insufficient padding will result in inaccurate calculation')
        if sample_plane.lower().strip() not in _valid_sample_planes:
            raise KeyError(f'sample plane must be one of {_valid_sample_planes}')
        if sample_size > 1:
            raise UserWarning('Coarse sampling may result in an undersampled point spread function')
        if opd_unit not in _opd_units:
            raise ValueError(f'opd_unit must be one of {opd_units}')

        if sample_plane.lower().strip() in _pupil_planes:
            self.sample_plane = 'pupil'
        else:
            self.sample_plane = 'image'

        self.wavelength = wavelength
        self.samples = samples
        self.padding = padding
        self.sample_size = sample_size
        self.opd_unit = opd_unit

    def __repr__(self):
        return (f'Simulation with properties:\n\t'
            f'wavelength: {self.wavelength}\n\t'
            f'samples: {self.samples}\n\t'
            f'padding: {self.padding}\n\t'
            f'sample plane: {self.sample_plane}\n\t'
            f'sample size: {self.sample_size}\n\t'
            f'opd unit: {self.opd_unit}')
