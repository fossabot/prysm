'''
A base pupil interface for different aberration models.
'''
from numpy import (
    nan, pi,
    arctan2, cos, sin,
    sqrt, exp,
    empty, ones,
    linspace, meshgrid,
    floor, isfinite,
    )
from numpy import power as npow
from matplotlib import pyplot as plt

from copy import deepcopy

from code6.util import share_fig_ax, rms
from code6.coordinates import cart_to_polar
from code6.units import (
    waves_to_microns, waves_to_nanometers,
    microns_to_waves, nanometers_to_waves,
    )

class Pupil(object):
    '''Pupil of an optical system

    Properties:
        slice_x: slice through the x axis of the pupil.  Returns (x,y)
                 data where x is the sample coordinate and y is the phase

        slice_y: slice through the y axis of the pupil.  Returns (x,y)
                 data where x is the sample coordinate and y is the phase

        pv: Peak-To-Valley wavefront error

        rms: Root Mean Square wavefront error

    Instance Methods:
        plot2d: Makes a 2D plot of the phase of the pupil.  Returns (fig, ax)

        plot_slice_xy: Makes a 1D plot of slices through X and Y of the phase
                       of the pupil.  Returns (fig, ax)

        interferogram: Makes an interferogram of the pupil.  Returns (fig, ax)

        build: Computes the phase and wavefunction of the pupil

        clip: Clips the pupil to a circular boundary

        merge: Merges this pupil with another, combining their OPD.  The two
            must be equally sampled.

    Private Instance Methods:
        _gengrid: generates the (x,y) and (rho,phi)

        _correct_phase_units: converts opd expressed in a given unit to waves

    Static Methods:
        none

    Notes:
        subclasses should implement a build() function and their own way of
            expressing OPD.
    '''
    def __init__(self, samples=128, epd=1, wavelength=0.55, opd_unit='$\lambda$'):
        '''Creates a new Pupil instance

        Args:
            samples (int): number of samples across pupil diameter

            wavelength (float): wavelength of light, in um

            epd: (float): diameter of the pupil, in mm

            opd_unit (string): unit OPD is expressed in.  One of:
                ($\lambda$, waves, $\mu m$, microns, um, nm , nanometers)

        Returns:
            Pupil.  A new Pupil instance
        '''

        self.samples          = samples
        self.wavelength       = wavelength
        self.opd_unit         = opd_unit
        self.phase = self.fcn = empty((samples, samples))
        self.unit             = linspace(-epd/2, epd/2, samples)
        self.unit_norm        = linspace(-1, 1, samples)
        self.sample_spacing   = self.unit[-1] - self.unit[-2]
        self.rho  = self.phi  = empty((samples, samples))
        self.center           = int(floor(samples/2))

        if self.opd_unit in ('$\lambda$', 'waves'):
            self._opd_unit = 'waves'
            self._opd_str = '$\lambda$'
        elif self.opd_unit in ('$\mu m$', 'microns', 'um'):
            self._opd_unit = 'microns'
            self._opd_str = '$\mu m$'
        elif self.opd_unit in ('nm', 'nanometers'):
            self._opd_unit = 'nanometers'
            self._opd_str = 'nm'
        else:
            raise ValueError('OPD must be expressed in waves, microns, or nm')

        self.build()
        self.clip()

    # quick-access slices, properties ------------------------------------------

    @property
    def slice_x(self):
        ''' Retrieves a slice through the X axis of the pupil
        '''
        return self.unit, self.phase[self.center, :]

    @property
    def slice_y(self):
        ''' Retrieves a slice through the Y axis of the pupil
        '''
        return self.unit, self.phase[:, self.center]

    @property
    def pv(self):
        ''' Returns the peak-to-valley wavefront error
        '''
        non_nan = isfinite(self.phase)
        return self.phase[non_nan].max() - self.phase[non_nan].min()

    @property
    def rms(self):
        ''' Returns the RMS wavefront error
        '''
        return rms(self.phase)

    # quick-access slices, properties ------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, fig=None, ax=None):
        ''' Creates a 2D plot of the phase error of the pupil

        Args:
            fig (pyplot.figure): Figure to draw plot in
            ax (pyplot.axis): Axis to draw plot in

        Returns:
            (pyplot.figure, pyplot.axis): Figure and axis containing the plot
        '''
        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(convert_phase(self.phase, self),
                       extent=[-epd/2, epd/2, -epd/2, epd/2],
                       cmap='Spectral',
                       interpolation='bicubic')
        fig.colorbar(im, label=f'OPD [{self._opd_str}]', ax=ax, fraction=0.046)
        ax.set(xlabel='Pupil X [mm]',
               ylabel='Pupil Y [mm]')
        return fig, ax

    def plot_slice_xy(self, fig=None, ax=None):
        ''' Creates a plot of slices through the X and Y axes of the pupil

        Args:
            fig (pyplot.figure): Figure to draw plot in
            ax (pyplot.axis): Axis to draw plot in

        Returns:
            (pyplot.figure, pyplot.axis): Figure and axis containing the plot
        '''
        u, x = self.slice_x
        _, y = self.slice_y

        fig, ax = share_fig_ax(fig, ax)

        x = convert_phase(x, self)
        y = convert_phase(y, self)

        ax.plot(u, x, lw=3, label='Slice X')
        ax.plot(u, y, lw=3, label='Slice Y')
        ax.set(xlabel=r'Pupil $\rho$ [mm]',
               ylabel=f'OPD [{self._opd_str}]')
        plt.legend()
        return fig, ax

    def interferogram(self, visibility=1, passes=2, fig=None, ax=None):
        ''' Creates an interferogram of the pupil.

        Args:
            visibility (float): Visibility of the interferogram
            passes (float): number of passes (double-pass, quadra-pass, etc.)
            fig (pyplot.figure): Figure to draw plot in
            ax (pyplot.axis): Axis to draw plot in

        Returns:
            (pyplot.figure, pyplot.axis): Figure and axis containing the plot
        '''
        fig, ax = share_fig_ax(fig, ax)
        plotdata = (visibility * sin(2 * pi * passes * self.phase))
        im = ax.imshow(plotdata,
                  extent=[-epd/2, epd/2, -epd/2, epd/2],
                  cmap='Greys_r',
                  interpolation='bicubic',
                  clim=(-1,1))
        fig.colorbar(im, label=r'Wrapped Phase [$\lambda$]')
        ax.set(xlabel='Pupil X [mm]',
               ylabel='Pupil Y [mm]')
        return fig, ax

    # meat 'n potatoes ---------------------------------------------------------

    def build(self):
        ''' Constructs a numerical model of a pupil.  The method should be overloaded by all
        subclasses to impart their unique mathematical models to the simulation.
        '''

        # build up the pupil
        self.phase = ones((self.samples, self.samples))
        self._correct_phase_units()
        self.fcn   = exp(1j * 2 * pi / self.wavelength * self.phase)

        # fill in the phase of the pupil
        self._gengrid()

        return self.unit, self.phase, self.fcn

    def clip(self):
        ''' Clips outside the circular boundary of the pupil
        '''
        self.phase[self.rho > 1] = nan
        self.fcn[self.rho > 1] = 0
        return self.phase, self.fcn

    def merge(self, pupil2):
        ''' Merges this pupil with another

        Args:
            pupil2 (Pupil): pupil with same sampling and OPD units as this one

        Returns:
            Pupil.  A new pupil with the OPD of both pupils combined
        '''
        return merge_pupils(self, pupil2)

    def _gengrid(self):
        '''Generates a uniform (x,y) grid and maps it to (rho,phi) coordinates for zernike eval
        '''
        x = y    = linspace(-1, 1, self.samples)
        xv, yv   = meshgrid(x,y)
        self.rho, self.phi = cart_to_polar(xv, yv)
        return self.rho, self.phi

    def _correct_phase_units(self):
        '''Converts an expression of OPD in a unit to waves
        '''
        if self._opd_unit == 'microns':
            self.phase *= waves_to_microns(self.wavelength)
        elif self._opd_unit == 'nanometers':
            self.phase *= waves_to_nanometers(self.wavelength)
        return self
    # meat 'n potatoes ---------------------------------------------------------

def convert_phase(array, pupil):
    '''Converts an OPD/phase map to have the same unit of expression as a pupil

    Args:
        array (numpy.ndarray): array of phase data
        pupil (code6.Pupil): a pupil to match the phase units to

    Returns:
        numpy.ndarray.  phase-corrected array.
    '''
    if pupil._opd_unit == 'microns':
        return array * microns_to_waves(pupil.wavelength)
    elif pupil._opd_unit == 'nanometers':
        return array * nanometers_to_waves(pupil.wavelength)
    else:
        return array

def merge_pupils(pupil1, pupil2):
    '''Merges the phase from two pupils and returns a new Pupil instance

    Args:
        pupil1 (Pupil): first pupil
        pupil2 (Pupil): second pupil

    Returns
        Pupil.  New pupil with merged phase
    '''
    if pupil1.sample_spacing != pupil2.sample_spacing or pupil1.samples != pupil2.samples:
        raise ValueError('Pupils must be identically sampled')

    # create a new pupil and copy Pupil1's dictionary into it
    props = deepcopy(pupil1.__dict__)
    retpupil = Pupil()
    retpupil.__dict__ = props

    retpupil.phase = pupil1.phase + pupil2.phase
    retpupil.fcn = exp(1j * 2 * pi / retpupil.wavelength * retpupil.phase)
    retpupil.clip()
    return retpupil
