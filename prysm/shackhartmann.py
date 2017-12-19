''' Shack Hartmann sensor modeling tools
'''
import numpy as np
from collections import deque

from matplotlib import pyplot as plt

from prysm.detector import bindown
from prysm.util import share_fig_ax


class ShackHartmann(object):
    ''' Shack Hartmann Wavefront Sensor object
    '''
    def __init__(self, sensor_size=(36, 24), pixel_pitch=3.75,
                 lenslet_pitch=375, lenslet_efl=2000, lenslet_fillfactor=0.9,
                 lenslet_array_shape='square', framebuffer=24):
        ''' Creates a new SHWFS object.

        Args:
            sensor_size (`iterable`): (x, y) sensor sizes in mm.

            pixel_pitch (`float`): center-to-center pixel spacing in um.

            lenslet_pitch (`float`): center-to-center spacing of lenslets in um.

            lenslet_efl (`float`): lenslet focal length, in microns.

            lenslet_fillfactor (`float`): portion of the sensor height filled
                by the lenslet array.  0.9 reserves 5% of the height on both
                the top and bottom of the array.

            lenslet_array_shape (`str`): `square` or `rectangular` --
                square will inscribe a square array within the detector area,
                rectangular will fill the detector area up to the fill factor.

            framebuffer (`int`): maximum number of frames of data to store.

        Returns:
            `ShackHartmann`: new Shack Hartmann wavefront sensor.
        '''
        # process lenslet array shape
        if lenslet_array_shape.lower() == 'square':
            self.lenslet_array_shape = 'square'
        elif lenslet_array_shape.lower() in ('rectangular', 'rect'):
            self.lenslet_array_shape = 'rectangular'

        # store data related to the silicon
        self.sensor_size = sensor_size
        self.pixel_pitch = pixel_pitch
        self.resolution = (int(sensor_size[0] // (pixel_pitch / 1e3)),
                           int(sensor_size[1] // (pixel_pitch / 1e3)))
        self.megapixels = self.resolution[0] * self.resolution[1] / 1e6

        # store lenslet metadata
        if self.lenslet_array_shape == 'square':
            xidx = 1
            sensor_extra_x = sensor_size[0] - sensor_size[1]
            if sensor_extra_x < 0:
                sensor_extra_y = sensor_size[1] - sensor_size[0]
                yshift = (sensor_extra_y / sensor_size[1]) / 2
                xshift = 0
            else:
                xshift = (sensor_extra_x / sensor_size[0]) / 2
                yshift = 0
        else:
            xidx = 0
            xshift = 0
            yshift = 0
        yidx = 1
        self.num_lenslets = (int(sensor_size[xidx] // (lenslet_pitch / 1e3) * lenslet_fillfactor),
                             int(sensor_size[yidx] // (lenslet_pitch / 1e3) * lenslet_fillfactor))
        self.total_lenslets = self.num_lenslets[0] * self.num_lenslets[1]
        self.lenslet_pitch = lenslet_pitch
        self.lenslet_efl = lenslet_efl

        # compute and store lenslet reference positions
        start_factor = (1 - lenslet_fillfactor) / 2
        end_factor = 1 - start_factor
        start_factor_x, end_factor_x = start_factor + xshift, end_factor - xshift
        start_factor_y, end_factor_y = start_factor + yshift, end_factor - yshift

        lenslet_start_x = start_factor_x * sensor_size[0] * 1e3  # factors of 1e3 convert mm to um
        lenslet_start_y = start_factor_y * sensor_size[1] * 1e3
        lenslet_end_x = end_factor_x * sensor_size[0] * 1e3
        lenslet_end_y = end_factor_y * sensor_size[1] * 1e3

        lenslet_pos_x = np.arange(lenslet_start_x, lenslet_end_x + lenslet_pitch, lenslet_pitch)
        lenslet_pos_y = np.arange(lenslet_start_y, lenslet_end_y + lenslet_pitch, lenslet_pitch)
        self.refx, self.refy = np.meshgrid(lenslet_pos_x, lenslet_pos_y)

        # initiate the frame buffer.
        self.buffer_depth = framebuffer
        self.captures = deque(maxlen=framebuffer)

    def __repr__(self):
        return ('Shack Hartmann sensor with: \n'
                f'\t({self.resolution[0]:}x{self.resolution[1]})px, {self.megapixels:.1f}MP CMOS\n'
                f'\t({self.num_lenslets[0]}x{self.num_lenslets[1]})lenslets, '
                f'{self.total_lenslets:1.0f} wavefront samples\n'
                f'\t{self.buffer_depth} frame buffer, currently storing {len(self.captures)} frames')

    def plot_reference_spots(self, fig=None, ax=None):
        ''' Create a plot of the reference positions of lenslets.

        Args:
            fig (`matplotlib.figure.Figure`): figure object to draw in.

            ax (`matplotlib.axes.Axis`): axis object to draw in.

        Returns:
            `tuple` containing:

                `matplotlib.figure.Figure`: figure containing the plot.

                `matplotlib.axes.Axis`: axis containing the plot.
        '''

        fig, ax = share_fig_ax(fig, ax)
        ax.scatter(self.refx / 1e3, self.refy / 1e3, c='k', s=4)
        ax.set(xlim=(0, self.sensor_size[0]), xlabel='Detector Position X [mm]',
               ylim=(0, self.sensor_size[1]), ylabel='Detector Position Y [mm]')
        return fig, ax

    def sample_wavefront(self, pupil):
        ''' Samples a wavefront, producing a shack-hartmann spot grid.

        Args:
            pupil (`Pupil`): a pupil object.

        Returns:
            TODO: return type

        Notes:
            Algorithm is as follows:
                1.  Compute the gradient of the wavefront.
                2.  Bindown the wavefront such that each sample in the output
                    corresponds to the local wavefront gradient at a lenslet.
                3.  Compute the x and y delta of each PSF in the image plane.
                4.  Shift each spot by the corresponding delta
        '''
        data = pupil.phase

        # compute the gradient
        dx, dy = np.gradient(data)

        # bin to the lenslet area
        nlenslets_y = self.num_lenslets[1]
        npupilsamples_y = pupil.samples
        npx_bin = npupilsamples_y // nlenslets_y
        dx_binned = bindown(dx, npx_bin)
        dy_binned = bindown(dy, npx_bin)

        pass


def psf_shift(lenslet_efl, dx, dy, mag=1):
    ''' Computes the shift of a PSF, in microns.

    Args:
        lenslet_efl (`microns`): EFL of lenslets.

        dx (`np.ndarray`): dx gradient of wavefront.

        dy (`np.ndarray`): dy gradient of wavefront.

        mag (`float`): magnification of the collimation system.

    Returns:
        `numpy.ndarray`: array of PSF shifts.

    '''
    pass
