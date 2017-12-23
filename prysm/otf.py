''' A base optical transfer function interface
'''
import numpy as np

from scipy import interpolate

from matplotlib import pyplot as plt

from prysm.mathops import fft2, fftshift, pi, sqrt
from prysm.psf import PSF
from prysm.fttools import forward_ft_unit
from prysm.util import correct_gamma, share_fig_ax
from prysm.coordinates import polar_to_cart


class MTF(object):
    ''' Modulation Transfer Function

    Properties:
        tan: slice along the X axis of the MTF object.

        sag: slice along the Y axis of the MTF object.

    Instance Methods:
        exact_polar: returns the exact MTF at a given set of frequency, azimuth
            pairs.  A list of frequencies can be given to evaluate along the X
            axis only.

        exact_xy: returns the exact MTF at a given X,Y frequency pair.  Returns
            a list of MTF values.

        plot2d: Makes a 2D plot of the MTF.  Returns (fig, ax)

        plot_tan_sag: Makes a plot of the tan/sag (x/y) MTF.  Returns (fig, ax)

    Private Instance Methods:
        _make_interp_function: generates an interpolation function for the MTF,
            and stores it in the class instance.

    Static Methods:
        from_psf: Generates an MTF object from a PSF.

        from_pupil: Generates an intermediate PSF object, and MTF from that PSF.

    '''
    def __init__(self, data, unit_x, unit_y=None):
        '''Creates an MTF object

        Args:
            data (`numpy.ndarray`): MTF values on 2D grid.

            unit_x (`numpy.ndarray`): array of x units.

            unit_y (`numpy.ndarray`): array of y units.

        Returns:
            `MTF`: a new :class:`MTF` instance.

        '''
        if unit_y is None:
            unit_y = unit_x
        self.data = data
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.samples_y, self.samples_x = data.shape
        self.center_x = self.samples_x // 2
        self.center_y = self.samples_y // 2

    # quick-access slices ------------------------------------------------------

    @property
    def tan(self):
        ''' Retrieves the tangential MTF.

        Notes:
            Assumes the object is extended in y.  If the object is extended
            along a different azimuth, this will not return the tangential MTF.
        '''
        return self.unit_x[self.center_x:], self.data[self.center_x:, self.center_y]

    @property
    def sag(self):
        ''' Retrieves the sagittal MTF.

        Notes:
            Assumes the object is extended in y.  If the object is extended along a different
            azimuth, this will not return the sagittal MTF.
        '''
        return self.unit_y[self.center_y:], self.data[self.center_x, self.center_y:]

    def exact_polar(self, freqs, azimuths=None):
        '''Retrieves the MTF at the specified frequency-azimuth pairs

        Args:
            freqs (`iterable`): radial frequencies to retrieve MTF for.

            azimuths (`iterable`): corresponding azimuths to retrieve MTF for.

        Returns:
            list: MTF at the given points.

        '''
        self._make_interp_function()

        # handle user-unspecified azimuth
        if azimuths is None:
            if type(freqs) in (int, float):
                # single azimuth
                azimuths = 0
            else:
                azimuths = [0] * len(freqs)
        # handle single azimuth, multiple freqs
        elif type(azimuths) in (int, float):
            azimuths = [azimuths] * len(freqs)

        azimuths = np.radians(azimuths)
        # handle single value case
        if type(freqs) in (int, float):
            x, y = polar_to_cart(freqs, azimuths)
            return float(self.interpf((x, y), method='linear'))

        outs = []
        for freq, az in zip(freqs, azimuths):
            x, y = polar_to_cart(freq, az)
            outs.append(float(self.interpf((x, y), method='linear')))
        return outs

    def exact_xy(self, x, y=None):
        '''Retrieves the MTF at the specified X-Y frequency pairs

        Args:
            x (iterable): X frequencies to retrieve the MTF at
            y (iterable): Y frequencies to retrieve the MTF at

        Returns:
            list: MTF at the given points

        '''
        self._make_interp_function()

        # handle user-unspecified azimuth
        if y is None:
            if type(x) in (int, float):
                # single azimuth
                y = 0
            else:
                y = [0] * len(x)

        # handle single value case
        if type(x) in (int, float):
            return float(self.interpf((x, y), method='linear'))

        outs = []
        for x, y in zip(x, y):
            outs.append(float(self.interpf((x, y), method='linear')))
        return outs
    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, log=False, max_freq=200, fig=None, ax=None):
        ''' Creates a 2D plot of the MTF.

        Args:
            log (`bool`): If true, plots on log scale.

            max_freq (`float`): Maximum frequency to plot to.  Axis limits will
                be ((-max_freq, max_freq), (-max_freq, max_freq)).

            fig (:class:`~matplotlib.pyplot.figure`): Figure to plot in.

            ax (:class:`~matplotlib.pyplot.axis`): Axis to plot in.

        Returns:
            `tuple` containing:
                fig (:class:`~matplotlib.pyplot.figure`): Figure containing the plot

                ax (:class:`~matplotlib.pyplot.axis`): Axis containing the plot.

        '''
        if log:
            fcn = 20 * np.log10(1e-24 + self.data)
            label_str = 'MTF [dB]'
            lims = (-120, 0)
        else:
            fcn = correct_gamma(self.data)
            label_str = 'MTF [Rel 1.0]'
            lims = (0, 1)

        left, right = self.unit_x[0], self.unit_x[-1]
        bottom, top = self.unit_y[0], self.unit_y[-1]

        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(fcn.T,
                       extent=[left, right, bottom, top],
                       origin='lower',
                       cmap='Greys_r',
                       interpolation='lanczo',
                       clim=lims)
        cb = fig.colorbar(im, label=label_str, ax=ax, fraction=0.046)
        cb.outline.set_edgecolor('k')
        cb.outline.set_linewidth(0.5)
        ax.set(xlabel=r'$\nu_x$ [cy/mm]',
               ylabel=r'$\nu_y$ [cy/mm]',
               xlim=(-max_freq, max_freq),
               ylim=(-max_freq, max_freq))
        return fig, ax

    def plot_tan_sag(self, max_freq=200, fig=None, ax=None, labels=('Tangential', 'Sagittal')):
        ''' Creates a plot of the tangential and sagittal MTF.

        Args:
            max_freq (`float`): Maximum frequency to plot to.  Axis limits will
                be ((-max_freq, max_freq), (-max_freq, max_freq)).

            fig (:class:`~matplotlib.pyplot.figure`): Figure to plot in.

            ax (:class:`~matplotlib.pyplot.axis`): Axis to plot in.

            labels (`iterable`): set of labels for the two lines that will be plotted.

        Returns:
            `tuple` containing:
                fig (:class:`~matplotlib.pyplot.figure`): Figure containing the plot.

                ax (:class:`~matplotlib.pyplot.axis`): Axis containing the plot.

        '''
        ut, tan = self.tan
        us, sag = self.sag

        fig, ax = share_fig_ax(fig, ax)
        ax.plot(ut, tan, label=labels[0], linestyle='-', lw=3)
        ax.plot(us, sag, label=labels[1], linestyle='--', lw=3)
        ax.set(xlabel='Spatial Frequency [cy/mm]',
               ylabel='MTF [Rel 1.0]',
               xlim=(0, max_freq),
               ylim=(0, 1))
        plt.legend(loc='lower left')
        return fig, ax

    # plotting -----------------------------------------------------------------

    # helpers ------------------------------------------------------------------

    def _make_interp_function(self):
        '''Generates an interpolation function for this instance of MTF, used to
            procure MTF at exact frequencies.`

        Args:
            none

        Returns:
            :class:`MTF`, this instance of an MTF object.

        '''
        if not hasattr(self, 'interpf'):
            self.interpf = interpolate.RegularGridInterpolator((self.unit_x, self.unit_y), self.data)

        return self

    @staticmethod
    def from_psf(psf):
        ''' Generates an MTF from a PSF.

        Args:
            psf (:class:`PSF`): PSF to compute an MTF from.

        Returns:
            :class:`MTF`: A new MTF instance.

        '''
        dat = abs(fftshift(fft2(psf.data)))
        unit_x = forward_ft_unit(psf.sample_spacing, psf.samples_x)
        unit_y = forward_ft_unit(psf.sample_spacing, psf.samples_y)
        return MTF(dat / dat[psf.center_x, psf.center_y], unit_x, unit_y)

    @staticmethod
    def from_pupil(pupil, efl, padding=1):
        ''' Generates an MTF from a pupil, given a focal length (propagation distance).

        Args:
            pupil (:class:`Pupil`): A pupil to propagate to a PSF, and convert to an MTF.

            efl (`float`): Effective focal length or propagation distance of the wavefunction.

            padding (`number`): Number of pupil widths to pad with on each side.

        Returns:
            :class:`MTF`: A new MTF instance.

        '''
        psf = PSF.from_pupil(pupil, efl=efl, padding=padding)
        return MTF.from_psf(psf)


def diffraction_limited_mtf(fno, wavelength=0.55, num_pts=128):
    ''' Gives the diffraction limited MTF for a circular pupil and the given parameters.

    Args:
        fno (`float`): f/# of the lens.

        wavelength (`float`): wavelength of light, in microns.

        num_pts (`int`): number of points in the output array.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: unit array, in cy/mm.

            `numpy.ndarray`: mtf array (rel. 1.0).
    '''
    normalized_frequency = np.linspace(0, 1, num_pts)
    extinction = 1 / (wavelength / 1000 * fno)
    mtf = (2 / pi) * \
          (np.arccos(normalized_frequency) - normalized_frequency *
           sqrt(1 - normalized_frequency ** 2))
    return normalized_frequency * extinction, mtf
