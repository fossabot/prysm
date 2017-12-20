''' Shack Hartmann sensor modeling tools
'''
from collections import deque

import numpy as np

from matplotlib import pyplot as plt

from prysm.mathops import pi
from prysm.detector import bindown
from prysm.util import share_fig_ax


class ShackHartmann(object):
    ''' Shack Hartmann Wavefront Sensor object
    '''
    def __init__(self, sensor_size=(36, 24), pixel_pitch=3.99999,
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
        # process lenslet array shape and lenslet offset
        if lenslet_array_shape.lower() == 'square':
            self.lenslet_array_shape = 'square'
        elif lenslet_array_shape.lower() in ('rectangular', 'rect'):
            self.lenslet_array_shape = 'rectangular'
        lenslet_shift = lenslet_pitch // 2

        # store data related to the silicon
        self.sensor_size = sensor_size
        self.pixel_pitch = pixel_pitch
        self.resolution = (int(sensor_size[0] // (pixel_pitch / 1e3)),
                           int(sensor_size[1] // (pixel_pitch / 1e3)))
        self.megapixels = self.resolution[0] * self.resolution[1] / 1e6

        # compute lenslet array shifts
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

        # store lenslet metadata - TODO: figure out why I need the - 1
        self.num_lenslets = (int(sensor_size[xidx] * lenslet_fillfactor // (lenslet_pitch / 1e3)) - 1,
                             int(sensor_size[yidx] * lenslet_fillfactor // (lenslet_pitch / 1e3)) - 1)
        self.total_lenslets = self.num_lenslets[0] * self.num_lenslets[1]
        self.lenslet_pitch = lenslet_pitch
        self.lenslet_efl = lenslet_efl
        self.lenslet_fno = self.lenslet_efl / self.lenslet_pitch

        # compute lenslet locations
        start_factor = (1 - lenslet_fillfactor) / 2
        end_factor = 1 - start_factor
        start_factor_x, end_factor_x = start_factor + xshift, end_factor - xshift
        start_factor_y, end_factor_y = start_factor + yshift, end_factor - yshift

        # factors of 1e3 convert mm to um, and round to 0.1nm to avoid machine precision errors
        lenslet_start_x = round(start_factor_x * sensor_size[0] * 1e3 + lenslet_shift, 4)
        lenslet_start_y = round(start_factor_y * sensor_size[1] * 1e3 + lenslet_shift, 4)
        lenslet_end_x = round(end_factor_x * sensor_size[0] * 1e3 - lenslet_shift, 4)
        lenslet_end_y = round(end_factor_y * sensor_size[1] * 1e3 - lenslet_shift, 4)

        lenslet_pos_x = np.linspace(lenslet_start_x, lenslet_end_x, self.num_lenslets[0])
        lenslet_pos_y = np.linspace(lenslet_start_y, lenslet_end_y, self.num_lenslets[0])
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
        ax.scatter(self.refx / 1e3, self.refy / 1e3, c='k', s=8)
        ax.set(xlim=(0, self.sensor_size[0]), xlabel='Detector Position X [mm]',
               ylim=(0, self.sensor_size[1]), ylabel='Detector Position Y [mm]',
               aspect='equal')
        return fig, ax

    def sample_wavefront(self, pupil, fig=None, ax=None):
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
        # grab the phase from the pupil and convert to units of length
        # TODO: remove assumption that phase has units of waves
        data = pupil.phase
        data = data * pupil.wavelength / 2 / pi

        # compute the gradient - TODO: see why gradient is dy,dx not dx,dy
        normalized_sample_spacing = 2 / pupil.samples
        dy, dx = np.gradient(data, normalized_sample_spacing, normalized_sample_spacing)

        # bin to the lenslet area
        nlenslets_y = self.num_lenslets[1]
        npupilsamples_y = pupil.samples

        npx_bin = npupilsamples_y // nlenslets_y
        print(type(npx_bin))

        dx_binned = bindown(dx, npx_bin, mode='avg')
        dy_binned = bindown(dy, npx_bin, mode='avg')
        shift_x, shift_y = psf_shift(self.lenslet_fno, dx_binned, dy_binned)

        fig, ax = share_fig_ax(fig, ax)
        ax.scatter(self.refx, self.refy, s=2, c='r')
        ax.scatter(self.refx - shift_x, self.refy - shift_y, s=4, c='k')
        ax.set(xlim=(0, self.sensor_size[0] * 1e3), xlabel='Detector Position X [mm]',
               ylim=(0, self.sensor_size[1] * 1e3), ylabel='Detector Position Y [mm]',
               aspect='equal')
        return fig, ax


def psf_shift(lenslet_efl, dx, dy, mag=1):
    ''' Computes the shift of a PSF, in microns.

    Args:
        lenslet_efl (`microns`): EFL of lenslets.

        dx (`np.ndarray`): dx gradient of wavefront.

        dy (`np.ndarray`): dy gradient of wavefront.

        mag (`float`): magnification of the collimation system.

    Returns:
        `numpy.ndarray`: array of PSF shifts.

    Notes:
        see eq. 12 of Chanan, "Principles of Wavefront Sensing and Reconstruction"
        delta = m * fl * grad(z)
        m is magnification of SH system
        fl is lenslet focal length
        grad(z) is the x, y gradient of the opd, z, which is expressed in
        physical units.
    '''
    coef = mag * lenslet_efl
    return coef * dx, coef * dy
