'''
A base pupil interface for different aberration models.
'''
from numpy import nan, pi, arctan2, cos, floor, sqrt, exp, empty, ones, linspace, meshgrid
from numpy import power as npow
from matplotlib import pyplot as plt

from code6.util import share_fig_ax
from code6.units import waves_to_microns, waves_to_nanometers, microns_to_waves, nanometers_to_waves

class Pupil(object):
    def __init__(self, samples=128, epd=1, autobuild=True, wavelength=0.55, opd_unit='$\lambda$'):
        self.samples          = samples
        self.wavelength       = wavelength
        self.opd_unit         = opd_unit
        self.phase = self.fcn = empty((samples, samples))
        self.unit             = linspace(-epd/2, epd/2, samples)
        self.unit_norm        = linspace(-1, 1, samples)
        self.sample_spacing   = self.unit[-1] - self.unit[-2]
        self.rho  = self.phi  = empty((samples, samples))
        self.center           = int(floor(samples/2))
        self.computed         = False
        self.rms              = 0
        self.PV               = 0

        if self.opd_unit in ('$\lambda$', 'waves'):
            self._opd_unit = 'waves'
            self._opd_str = '$\lambda$'
        elif self.opd_unit in ('$\mu m', 'microns', 'um'):
            self._opd_unit = 'microns'
            self._opd_str = '$\mu m$'
        elif self.opd_unit in ('nm', 'nanometers'):
            self._opd_unit = 'nanometers'
            self._opd_str = 'nm'
        else:
            raise ValueError('OPD must be expressed in waves, microns, or nm')

        if autobuild:
            self.build()
            self.clip()
            self.computed = True

    # quick-access slices, properties ------------------------------------------

    @property
    def slice_x(self):
        '''
        Retrieves a slice through the X axis of the pupil
        '''
        return self.unit, self.phase[self.center, :]

    @property
    def slice_y(self):
        '''
        Retrieves a slice through the Y axis of the pupil
        '''
        return self.unit, self.phase[:, self.center]

    # quick-access slices, properties ------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, fig=None, ax=None):
        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(convert_phase(self.phase, self),
                       extent=[-1, 1, -1, 1],
                       cmap='Spectral',
                       interpolation='bicubic')
        fig.colorbar(im, label=f'OPD [{self._opd_str}]', ax=ax, fraction=0.046)
        ax.set(xlabel='Normalized Pupil X [a.u.]',
               ylabel='Normalized Pupil Y [a.u.]')
        return fig, ax

    def plot_slice_xy(self):
        u, x = self.slice_x
        _, y = self.slice_y

        fig, ax = plt.subplots()

        x = convert_phase(x, self)
        y = convert_phase(y, self)

        ax.plot(u, x, lw=3, label='Slice X')
        ax.plot(u, y, lw=3, label='Slice Y')
        ax.set(xlabel=r'Pupil $\rho$ [mm]',
               ylabel=f'OPD [{self._opd_str}]')
        plt.legend()
        return fig, ax

    def interferogram(self, visibility=1, fig=None, ax=None):
        phase = convert_phase(self.phase, self)
        fig, ax = share_fig_ax(fig, ax)
        plotdata = (0.5 + 0.5 * visibility * cos(2 * pi * phase))
        im = ax.imshow(plotdata,
                  extent=[-1, 1, -1, 1],
                  cmap='Greys_r',
                  interpolation='bicubic',
                  clim=(-1,1))
        fig.colorbar(im, label=f'Wrapped Phase [{self._opd_str}]')
        ax.set(xlabel='Normalized Pupil X [a.u.]',
               ylabel='Normalized Pupil Y [a.u.]')
        return fig, ax

    # meat 'n potatoes ---------------------------------------------------------

    def build(self):
        '''
        Constructs a numerical model of an exit pupil.  The method should be overloaded
        by all subclasses to impart their unique mathematical models to the simulation.
        '''

        # build up the pupil
        self.phase = ones((self.samples, self.samples))
        self._correct_phase_units()
        self.fcn   = exp(1j * 2 * pi / self.wavelength * self.phase)

        # fill in the phase of the pupil
        self._gengrid()

        return self.unit, self.phase, self.fcn

    def clip(self):
        '''
        # clip outside the circular boundary of the pupil
        '''
        self.phase[self.rho > 1] = nan
        self.fcn[self.rho > 1] = 0
        return self.phase, self.fcn

    def _gengrid(self):
        x = y    = linspace(-1, 1, self.samples)
        xv, yv   = meshgrid(x,y)
        self.rho = sqrt(npow(xv,2) + npow(yv,2))
        self.phi = arctan2(yv, xv)
        return self.rho, self.phi

    def _correct_phase_units(self):
        if self._opd_unit == 'microns':
            self.phase *= waves_to_microns(self.wavelength)
        elif self._opd_unit == 'nanometers':
            self.phase *= waves_to_nanometers(self.wavelength)
        return self
    # meat 'n potatoes ---------------------------------------------------------

def convert_phase(array, pupil):
    if pupil._opd_unit == 'microns':
        return array * microns_to_waves(pupil.wavelength)
    elif pupil._opd_unit == 'nanometers':
        return array * nanometers_to_waves(pupil.wavelength)
    else:
        return array