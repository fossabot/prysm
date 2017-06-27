'''
A base pupil interface for different aberration models.
'''
from numpy import nan, pi, arctan2, floor, sqrt, exp, empty, ones, linspace, meshgrid
from matplotlib import pyplot as plt

class Pupil(object):
    def __init__(self, samples=512, epd=1):
        self.samples          = samples
        self.phase = self.fcn = empty((samples, samples))
        self.unit             = linspace(-epd/2, epd/2, samples)
        self.unit_norm        = linspace(-1, 1, samples)
        self.sample_spacing   = self.unit[-1] - self.unit[-2]
        self.rho  = self.phi  = empty((samples, samples))
        self.center           = int(floor(samples/2))
        self.computed         = False

    # quick-access slices ------------------------------------------------------

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

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, opd_unit='$\lambda$'):
        fig, ax = plt.subplots()
        im = ax.imshow(self.phase,
                       extent=[-1, 1, -1, 1],
                       cmap='Spectral',
                       interpolation='bicubic')
        fig.colorbar(im, label=f'OPD [{opd_unit}]')
        ax.set(xlabel='Normalized Pupil X [a.u.]',
               ylabel='Normalized Pupil Y [a.u.]')
        plt.show()
        return fig, ax

    def plot_slice_xy(self, opd_unit='$\lambda$'):
        fig, ax = plt.subplots()
        u, x = self.pupil_slice_x
        _, y = self.pupil_slice_y
        ax.plot(u, x, lw=3, label='Slice X')
        ax.plot(u, y, lw=3, label='Slice Y')
        ax.set(xlabel=r'Pupil $\rho$ [mm]',
               ylabel=f'OPD [{opd_unit}]')
        plt.legend()
        plt.show()
        return fig, ax

    # meat 'n potatoes ---------------------------------------------------------
    def build(self, wavelength=0.5):
        '''
        Constructs a numerical model of an exit pupil.  The method should be overloaded
        by all subclasses to impart their unique mathematical models to the simulation.
        '''

        # build up the pupil
        self.phase = ones((self.samples, self.samples))
        self.fcn   = exp(1j * 2 * pi / wavelength * self.phase)

        # fill in the phase of the pupil
        x = y    = self.unit_norm
        xv, yv   = meshgrid(x,y)
        self.rho = sqrt(xv**2, yv**2)
        self.phi = arctan2(yv, xv)

        return self.unit, self.phase, self.fcn

    def clip(self):
        '''
        # clip outside the circular boundary of the pupil
        '''
        self.phase[self.rho > 1] = nan
        self.fcn[self.rho > 1] = 0
        return self.phase, self.fcn

    # meat 'n potatoes ---------------------------------------------------------