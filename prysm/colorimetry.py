''' optional tools for colorimetry, wrapping the color-science library, see:
    http://colour-science.org/
'''
import numpy as np

try:
    import colour
except ImportError:
    # Spectrum objects can be used without colour
    pass

from prysm.util import share_fig_ax

class Spectrum(object):
    ''' Representation of a spectrum of light.
    '''
    def __init__(self, wavelengths, values):
        ''' makes a new Spectrum instance.

        Args:
            wavelengths (`numpy.ndarray`): wavelengths values correspond to.
                units of nanometers.

            values (`numpy.ndarray`): values associated with the wavelengths.
                arbitrary units.

        Returns:
            `Spectrum`: new Spectrum object.

        '''
        self.wavelengths = np.asarray(wavelengths)
        self.values = np.asarray(values)
        self.meta = dict()

    def plot(self, xrange=None, yrange=(0, 100), fig=None, ax=None):
        ''' Plots the spectrum.

        Args:
            xrange (`iterable`): pair of lower and upper x bounds.

            yrange (`iterable`): pair of lower and upper y bounds.

            fig (`matplotlib.figure.Figure`): figure to plot in.

            ax (`matplotlib.axes.Axis`): axis to plot in.

        Returns:
            `tuple` containing:

                `matplotlib.figure.Figure`: figure containign the plot.

                `matplotlib.axes.Axis`: axis containing the plot.

        '''

        fig, ax = share_fig_ax(fig, ax)

        ax.plot(self.wavelengths, self.values)
        ax.set(xlim=xrange, xlabel=r'Wavelength $\lambda$ [nm]',
               ylim=yrange, ylabel='Transmission [%]')

        return fig, ax

class CIEXYZ(object):
    ''' CIE XYZ 1931 color coordinate system.
    '''

    def __init__(self, x, y, z):
        ''' Creates a new CIEXYZ object.

        Args:
            x (`numpy.ndarray`): array of x coordinates.

            y (`numpy.ndarray`): array of y coordinates.

            z (`numpy.ndarray`): z unit array.

        Returns:
            `CIEXYZ`: new CIEXYZ object.

        '''
        self.x = x
        self.y = y
        self.z = z

    def to_xy(self):
        ''' Returns the x, y coordinates
        '''
        check_colour()
        x, y = colour.XYZ_to_xy((self.x, self.y, self.z))
        return x, y

    @staticmethod
    def from_spectrum(spectrum):
        ''' computes XYZ coordinates from a spectrum

        Args:
            spectrum (`Spectrum`): a Spectrum object.

        Returns:
            `CIEXYZ`: a new CIEXYZ object.

        '''
        # we need colour
        check_colour()

        # convert to a spectral power distribution
        spectrum = normalize_spectrum(spectrum)
        spectrum_dict = dict(zip(spectrum.wavelengths, spectrum.values))
        spd = colour.SpectralPowerDistribution('', spectrum_dict)

        # map onto correct wavelength pts
        standard_wavelengths = colour.SpectralShape(start=360, end=780, interval=10)
        xyz = colour.colorimetry.spectral_to_XYZ(spd.align(standard_wavelengths))
        return CIEXYZ(*xyz)

class CIELUV(object):
    ''' CIE 1976 color coordinate system.

    Notes:
        This is the CIE L* u' v' system, not LUV.
    '''

    def __init__(self, u, v, l=None):
        ''' Creates a new CIELUV instance

        Args:
            u (`float`): u coordinate

            v (`float`): v coordinate

            l (`float): l coordinate

        Returns:
            `CIELUV`: new CIELIV instance.

        '''
        self.u = u
        self.v = v
        self.l = l

    @staticmethod
    def from_XYZ(ciexyz=None, x=None, y=None, z=None):
        ''' Computes CIEL*u'v' coordinates from XYZ coordinate.

        Args:
            ciexyz (`CIEXYZ`): CIEXYZ object holding X,Y,Z coordinates.

            x (`float`): x coordinate.

            y (`float`): y coordinate.

            z (`float`): z coordinate.

        Returns:
            `CIELUV`: new CIELUV object.

        Notes:
            if given ciexyz object, x, y, and z are not used.  If given x, y, z,
            then ciexyz object is not needed.

        '''
        if ciexyz is not None:
            x, y, z = ciexyz.x, ciexyz.y, ciexyz.z
        elif x is None and y is None and z is None:
            raise ValueError('all values are None')

        l, u, v = colour.XYZ_to_Luv((x, y, z))
        return CIELUV(u, v, l)

    @staticmethod
    def from_spectrum(spectrum):
        ''' converts a spectrum to CIELUV coordinates.

        Args:
            spectrum (`Spectrum`): spectrum object to convert.

        Returns:
            `CIELUV`: new CIELUV object.

        '''
        xyz = CIEXYZ.from_spectrum(spectrum)
        return CIELUV.from_XYZ(xyz)

def normalize_spectrum(spectrum):
    ''' Normalizes a spectrum to have unit peak within the visible band.
    Args:
        spectrum (`Spectrum`): object with iterable wavelength, value fields.

    Returns:
        `Spectrum`: new spectrum object.

    '''
    wvl, vals = spectrum.wavelengths, spectrum.values
    low, high = np.searchsorted(wvl, 400), np.searchsorted(wvl, 700)
    vis_values_max = vals[low:high].max()
    return Spectrum(wvl, vals/vis_values_max)

def check_colour():
    ''' Checks if colour is available, raises if not.
    '''
    if 'colour' not in globals(): # or in locals
            raise ImportError('prysm colorimetry requires the colour package, '
                              'see http://colour-science.org/installation-guide/')
