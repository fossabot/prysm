'''
A base pupil interface for different aberration models.
'''
from numpy import nan, pi, arctan2, cos, floor, sqrt, exp, empty, ones, linspace, meshgrid
from numpy import power as npow
from matplotlib import pyplot as plt

from code6.util import share_fig_ax

class Pupil(object):
    def __init__(self, samples=128, epd=1, autobuild=True, wavelength=0.55):
        self.samples          = samples
        self.wavelength       = wavelength
        self.phase = self.fcn = empty((samples, samples))
        self.unit             = linspace(-epd/2, epd/2, samples)
        self.unit_norm        = linspace(-1, 1, samples)
        self.sample_spacing   = self.unit[-1] - self.unit[-2]
        self.rho  = self.phi  = empty((samples, samples))
        self.center           = int(floor(samples/2))
        self.computed         = False
        self.rms              = 0
        self.PV               = 0

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

    def plot2d(self, opd_unit='$\lambda$', fig=None, ax=None):
        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(self.phase,
                       extent=[-1, 1, -1, 1],
                       cmap='Spectral',
                       interpolation='bicubic')
        fig.colorbar(im, label=f'OPD [{opd_unit}]', ax=ax, fraction=0.046)
        ax.set(xlabel='Normalized Pupil X [a.u.]',
               ylabel='Normalized Pupil Y [a.u.]')
        return fig, ax

    def plot_slice_xy(self, opd_unit='$\lambda$'):
        fig, ax = plt.subplots()
        u, x = self.slice_x
        _, y = self.slice_y
        ax.plot(u, x, lw=3, label='Slice X')
        ax.plot(u, y, lw=3, label='Slice Y')
        ax.set(xlabel=r'Pupil $\rho$ [mm]',
               ylabel=f'OPD [{opd_unit}]')
        plt.legend()
        return fig, ax

    def interferogram(self, visibility=1, opd_unit='$\lambda$', fig=None, ax=None):
        fig, ax = share_fig_ax(fig, ax)
        plotdata = (0.5 + 0.5 * visibility * cos(2 * pi * self.phase))
        im = ax.imshow(plotdata,
                  extent=[-1, 1, -1, 1],
                  cmap='Greys_r',
                  interpolation='bicubic',
                  clim=(-1,1))
        fig.colorbar(im, label=f'Wrapped Phase [{opd_unit}]')
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
    # meat 'n potatoes ---------------------------------------------------------
