'''
A base optical transfer function interface
'''
import numpy as np
from numpy import floor
from numpy.fft import fft2, fftshift, ifftshift

from matplotlib import pyplot as plt

class MTF(object):
    def __init__(self, data, unit):
        # dump inputs into class instance
        self.data = data
        self.unit = unit
        self.samples = len(unit)
        self.center = int(floor(self.samples/2))

    # quick-access slices ------------------------------------------------------

    @property
    def tan(self):
        '''
        Retrieves the tangential MTF
        '''
        return self.unit[self.center:-1], self.data[self.center, self.center:-1]

    @property
    def sag(self):
        '''
        Retrieves the sagittal MTF
        '''
        return self.unit[self.center:-1], self.data[self.center:-1, self.center]

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, log=False):
        if log:
            fcn = 20 * np.log10(1e-24 + self.data)
            label_str = 'MTF [dB]'
            lims = (-120, 0)
        else:
            fcn = self.data
            label_str = 'MTF [Rel 1.0]'
            lims = (0, 1)

        left, right = self.unit[0], self.unit[-1]

        fig, ax = plt.subplots()
        im = ax.imshow(fcn,
                       extent=[left, right, left, right],
                       cmap='Greys_r',
                       interpolation='bicubic',
                       clim=lims)
        fig.colorbar(im, label=label_str)
        ax.set(xlabel='Spatial Frequency X [cy/mm]',
               ylabel='Spatial Frequency Y [cy/mm]',
               xlim=(-200,200),
               ylim=(-200,200))
        return fig, ax

    def plot_tan_sag(self):
        u, tan = self.tan
        _, sag = self.sag

        fig, ax = plt.subplots()
        ax.plot(u, tan, label='Tangential', linestyle='-', lw=3)
        ax.plot(u, sag, label='Sagittal', linestyle='--', lw=3)
        ax.set(xlabel='Spatial Frequency [cy/mm]',
               ylabel='MTF [Rel 1.0]',
               xlim=(0,200),
               ylim=(0,1))
        plt.legend(loc='lower left')
        return fig, ax

    # plotting -----------------------------------------------------------------

    @staticmethod
    def from_psf(psf):
        dat = abs(fftshift(fft2(psf.data)))
        f_s = int(floor(psf.samples / 2))
        unit = 1 / (psf.sample_spacing / 1e3) * range(-f_s, f_s) / psf.samples
        return MTF(dat/dat[f_s,f_s], unit)
