''' Allows first-order calculation for zoom lenses with multiple elements.
'''
import numpy as np

from prysm.util import share_fig_ax

class TwoElementZoom(object):
    ''' Represents a two-element infinite conjugate zoom
    '''
    def __init__(self, f1, f2, efl_long, efl_short):
        self.f1 = f1
        self.f2 = f2
        self.long = efl_long
        self.short = efl_short
        self.ratio = efl_long/efl_short

    def l2_image_position(self, separation):
        ''' Uses the separation of the two elements to compute the second lens
            position.

        Args:
            separation (`float`): separation of lens 1 and lens 2

        Returns:
            float: position of the second lens.

        '''
        t = separation
        return ((self.f1 - t) * self.f2) /\
               ((self.f1 - t) + self.f2)

    def efl_to_separation(self, efl):
        ''' Computes the separation of the two elements from the system focal
            length.

        Args:
            efl (`float`): system EFL.

        Returns:
            float. the separation of the two lenses.

        '''
        return self.f1 + self.f2 - (self.f1*self.f2)/efl

    def _lens1pos(self, num_pts):
        ''' Computes the position of element 1 for the given number of points
            throughout the zoom range.

        Args:
            num_pts (`int`): number of points to compute the position for.

        Returns:
            `numpy.ndarray`. vector of lens 1 positions.

        '''
        efls = np.linspace(self.short, self.long, num_pts)
        seps = self.efl_to_separation(efls)
        s2p = self.l2_image_position(seps)
        return seps+s2p

    def _lens2pos(self, num_pts):
        ''' Compute the position of element 2 for the given number of points
            throughout the zoom range.

        Args:
            num_pts (`int`): number of points to compute the position for.

        Returns:
            `numpy.ndarray`. vector of lens 2 positions.

        '''
        efls = np.linspace(self.short, self.long, num_pts)
        seps = self.efl_to_separation(efls)
        s2p = self.l2_image_position(seps)
        return s2p

    def lenspos(self, num_pts=100):
        ''' Computes the position of the two elements as a function of the
            system focal length for the given number of points.

        Args:
            num_pts (`int`): number of points to compute element position for.

        Returns:
            `tuple` containing:

                `numpy.ndarray`.  ndarray containing element 1 position.

                `numpy.ndarray`.  ndarray containing element 2 position.

        '''
        return self._lens1pos(num_pts), self._lens2pos(num_pts)

    def plot_lenspos(self, num_pts=100, ylims=None, fig=None, ax=None):
        ''' Plots the position of the two elements as a function of system EFL

        Args:
            num_pts (`int`): number of points in the plot.

            ylims (`tuple`): y axis limits of the plot.

            fig (`matplotlib.pyplot.figure`): figure containing the output plot.

            ax (`matplotlib.pyplot.axis`): axis containing the output plot.

        Returns:
            `tuple` containing:

                `matplotlib.pyplot.figure`. figure containing the plot.

                `matplotlib.pyplot.axis`. axis containing the plot.

        '''
        efls = np.linspace(self.short, self.long, num_pts)
        l1p, l2p = self.lenspos(num_pts)

        fig, ax = share_fig_ax(fig, ax)

        ax.plot(efls, l1p, label='Lens 1', lw=3)
        ax.plot(efls, l2p, label='Lens 2', lw=3)
        ax.set(xlim=(self.short, self.long), xlabel='Focal Length [mm]',
               ylim=ylims, ylabel='Distance from Image [mm]',
               title=f'{self.short}-{self.long}mm lens element positions')
        ax.legend()

        return fig, ax