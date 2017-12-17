''' Shack Hartmann sensor modeling tools
'''
import numpy as np


class ShackHartmann(object):
    ''' Shack Hartmann Wavefront Sensor object
    '''
    def __init__(self, sensor_size=(24, 36), pixel_pitch=4, lenslet_pitch=350):
        ''' Creates a new SHWFS object.

        Args:
            sensor_size (`iterable`): (x, y) sensor sizes in mm.

            pixel_pitch (`float`): center-to-center pixel spacing in um.

            lenslet_pitch (`float`): center-to-center spacing of lenslets in um.

        Returns:
            `ShackHartmann`: new Shack Hartmann wavefront sensor.
        '''
        self.sensor_size = sensor_size
        self.pixel_pitch = pixel_pitch
        self.resolution = (sensor_size[0] // (pixel_pitch / 1e3),
                           sensor_size[1] // (pixel_pitch / 1e3))
        self.megapixels = self.resolution[0] * self.resolution[1] / 1e6

        self.lenslet_pitch = lenslet_pitch
        self.num_lenslets = (sensor_size[0] / lenslet_pitch * 1e3,
                             sensor_size[1] / lenslet_pitch * 1e3)
        self.total_lenslets = self.num_lenslets[0] * self.num_lenslets[1]

    def __repr__(self):
        return ('Shack Hartmann Sensor with specs: \n'
                f'\t({self.resolution[0]}x{self.resolution[1]})px, {self.megapixels:.1f}MP CMOS\n'
                f'\t({self.num_lenslets[0]}x{self.num_lenslets[1]})lenslets,'
                f'{self.total_lenslets:1.0f} wavefront samples')

    def sample_wavefront(self):
        ''' Samples a wavefront, producing a shack-hartmann spot grid.
        '''
        pass
