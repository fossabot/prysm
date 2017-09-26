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
        ax.legend()
        ax.set(xlim=(self.short, self.long), xlabel='Focal Length [mm]',
               ylim=ylims, ylabel='Distance from Image [mm]',
               title=f'{self.short}-{self.long}mm lens element positions')

        return fig, ax

class TwoElementZoomFinite(object):
    ''' Represents a two element finite conjugate zoom lens.
    '''
    def __init__(self, efl1, efl2, objdist, mlow, mhigh):
        self.f1 = efl1
        self.f2 = efl2
        self.objdist = objdist
        self.low = mlow
        self.high = mhigh

    def _separation(self, mag, root='positive'):
        ''' Returns the lens separation solution with the given root

        Args:
            mag (`float`): magnification of the system

            root (`string`): which root to return

        Returns:
            float. separation of the two lens elements.

        '''
        M, f1, f2, L = mag, self.f1, self.f2, self.objdist
        d = L**2 - 4*(L*(f1+f2) + (M-1)**2 *f1*f2/M)
        if root.lower() in ('p', 'pos', 'positive'):
            return (L + np.sqrt(d))/2
        else:
            return (L - np.sqrt(d))/2

    def _genpositions(self, mag, num_pts=100, root='p'):
        M, L, f1 = mag, self.objdist, self.f1
        t = self._separation(M, root=root)
        sp = ((M-1)*t+L)/((M-1)-M*t/f1)
        lens1pos = L + sp
        lens2pos = L + sp - t
        objpos = [L] * len(M)
        return lens1pos, lens2pos, objpos

    def plot_posroot_soln(self, num_pts=100, ylims=None, fig=None, ax=None):
        M = np.linspace(self.low, self.high, num_pts)
        l1p, l2p, op = self._genpositions(M, num_pts, root='p')

        fig, ax = share_fig_ax(fig, ax)
        highlight_zero((l1p, l2p, op), ax=ax)

        ax.plot(M, l1p, lw=3, label='Lens 1')
        ax.plot(M, l2p, lw=3, label='Lens 2')
        ax.plot(M, op, lw=1.5, c='k', label='Object')
        ax.legend()
        ax.set(xlim=(self.low, self.high), xlabel='Magnification [-]',
               ylim=ylims, ylabel='Distance from Image [mm]',
               title=f'{self.low}-{self.high}x Zoom lens')

        return fig, ax

    def plot_negroot_soln(self, num_pts=100, ylims=None, fig=None, ax=None):
        M = np.linspace(self.low, self.high, num_pts)
        l1p, l2p, op = self._genpositions(M, num_pts, root='n')

        fig, ax = share_fig_ax(fig, ax)
        highlight_zero((l1p, l2p, op), ax=ax)

        ax.plot(M, l1p, lw=3, label='Lens 1')
        ax.plot(M, l2p, lw=3, label='Lens 2')
        ax.plot(M, op, lw=3, c='k', label='Object')
        ax.legend()
        ax.set(xlim=(self.low, self.high), xlabel='Magnification [-]',
               ylim=ylims, ylabel='Distance from Image [mm]',
               title=f'{self.low}-{self.high}x Zoom lens')

        return fig, ax

    def plot_both_solns(self, fig=None, axs=None):
        fig, axs = share_fig_ax(fig, axs, numax=2)

        self.plot_posroot_soln(fig=fig, ax=axs[0])
        self.plot_negroot_soln(fig=fig, ax=axs[1])

        fig.tight_layout()
        return fig, axs

def highlight_zero(data, ax):
    try:
        # the iterable case with multiple datas.
        # hasattr will error if the first element of data isn't an iterable;
        # because data is either a list or np array, we need two depths of
        # iterability for this to behave as expected.
        hasattr(data[0], '__iter__')
        drawline = False
        for datum in data:
            if min(datum) < 0:
                drawline = True

        if drawline is True:
            ax.axhline(0, c='k', ls=':')
    except TypeError:
        if min(data) < 0:
            ax.axhline(0, c='k', ls=':')