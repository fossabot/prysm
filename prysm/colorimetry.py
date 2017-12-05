''' optional tools for colorimetry, wrapping the color-science library, see:
    http://colour-science.org/
'''
import warnings

import numpy as np

from matplotlib import pyplot as plt

try:
    import colour
except ImportError:
    # Spectrum objects can be used without colour
    pass

from prysm.util import share_fig_ax, correct_gamma
from prysm.geometry import generate_mask

# some CIE constants
CIE_K = 24389 / 27
CIE_E = 216 / 24389

# sRGB conversion matrix
XYZ_to_sRGB_mat_D65 = np.asarray([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252]
])
XYZ_to_sRGB_mat_D50 = np.asarray([
    [3.1338561, -1.6168667, -0.4906146],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0719453, -0.2289914, 1.4052427],
])

# Adobe RGB 1998 matricies
XYZ_to_AdobeRGB_mat_D65 = np.asarray([
    [2.0413690, -0.5649464, -0.3446944],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0134474, -0.1183897, 1.0154096],
])
XYZ_to_AdobeRGB_mat_D50 = np.asarray([
    [1.9624274, -0.6105343, -0.3413404],
    [-0.9787684, 1.9161415, 0.0334540],
    [0.0286869, -0.1406752, 1.3487655],
])
COLOR_MATRICIES = {
    'sRGB': {
        'D65': XYZ_to_sRGB_mat_D65,
        'D50': XYZ_to_sRGB_mat_D50,
    },
    'AdobeRGB': {
        'D65': XYZ_to_AdobeRGB_mat_D65,
        'D50': XYZ_to_AdobeRGB_mat_D50,
    },
}

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
        return colour.XYZ_to_xy((self.x, self.y, self.z))

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
        spd = spd_from_spectrum(spectrum)

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

    def to_uv(self):
        ''' Returns the u, v coordinates.
        '''
        return colour.Luv_to_uv((self.l, self.u, self.v))

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

def spd_from_spectrum(spectrum):
    ''' converts a spectrum to a colour spectral power distribution object.

    Args:
        spectrm (`Spectrum`): spectrum object to conver to spd.

    Returns:
        `SpectralPowerDistribution`: colour SPD object.

    '''
    spectrum_dict = dict(zip(spectrum.wavelengths, spectrum.values))
    return colour.SpectralPowerDistribution('', spectrum_dict)

def check_colour():
    ''' Checks if colour is available, raises if not.
    '''
    if 'colour' not in globals(): # or in locals
            raise ImportError('prysm colorimetry requires the colour package, '
                              'see http://colour-science.org/installation-guide/')

def cie_1976_plot(xlim=(0, 0.7), ylim=None, samples=200, fig=None, ax=None):
    ''' Creates a CIE 1976 plot.

    Args:

        xlim (`iterable`): left and right bounds of the plot.

        ylim (`iterable`): lower and upper bounds of the plot.  If `None`,
            the y bounds will be chosen to match the x bounds.

        samples (`int`): number of 1D samples within the region of interest,
            total pixels will be samples^2.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the plot.

            `matplotlib.axes.axis`: axis containing the plot.

    '''
    # ignore runtime warnings -- invalid values in power for some u,v -> sRGB values
    warnings.simplefilter('ignore', RuntimeWarning)

    # duplicate xlim if ylim not set
    if ylim is None:
        ylim = xlim

    # create lists of wavelengths and map them to uv,
    # a reduced set for a faster mask and
    # an equally spaced set for annotation
    wvl_line = np.arange(400, 700, 2)
    wvl_line_uv = XYZ_to_uv(colour.wavelength_to_XYZ(wvl_line))

    wvl_annotate = [400, 440, 450, 470, 480, 490,
                    500, 510, 520, 540, 560, 570, 580, 590,
                    600, 610, 630, 700]
    wvl_annotate_uv = XYZ_to_uv(colour.wavelength_to_XYZ(wvl_annotate))

    wvl_mask = [400, 430, 460, 465, 470, 475, 480, 485, 490, 495,
                500, 505, 510, 515, 520, 525, 530, 535, 570, 700]

    wvl_mask_XYZ = colour.wavelength_to_XYZ(wvl_mask)
    wvl_mask_uv = XYZ_to_uv(wvl_mask_XYZ)
    wvl_pts = wvl_mask_uv * samples / np.array([xlim[1], ylim[1]])
    wvl_mask = generate_mask(wvl_pts, samples)

    # make equally spaced u,v coordinates on a grid
    u = np.linspace(xlim[0], xlim[1], samples)
    v = np.linspace(ylim[0], ylim[1], samples)
    uu, vv = np.meshgrid(u, v)

    # set values outside the horseshoe to a safe value that won't blow up

    # stack u and v for vectorized computations
    uuvv = np.stack((vv, uu), axis=2)

    xy = Luv_uv_to_xy(uuvv)
    xyz = xy_to_XYZ(xy)
    dat = XYZ_to_sRGB(xyz)
    # normalize and clip
    dat /= np.max(dat, axis=2)[..., np.newaxis]
    dat = np.clip(dat, 0, 1)

    # now make an alpha/transparency mask to hide the background
    # and flip u,v axes because of column-major symantics
    alpha = np.ones((samples, samples)) * wvl_mask
    dat = np.swapaxes(np.dstack((dat, alpha)), 0, 1)

    # lastly, duplicate the lowest wavelength so that the boundary line is closed
    wvl_line_uv = np.vstack((wvl_line_uv, wvl_line_uv[0, :]))

    fig, ax = share_fig_ax(fig, ax)
    ax.imshow(dat,
              extent=[*xlim, *ylim],
              interpolation='bilinear',
              origin='lower')
    ax.plot(wvl_line_uv[:, 0], wvl_line_uv[:, 1], ls='-', c='0.25', lw=2)
    for wvl, pts in zip(wvl_annotate, wvl_annotate_uv):
        ax.annotate(wvl, xy=pts)

    ax.set(xlim=(-0.025, 0.65), xlabel='CIE u\'',
           ylim=(0, 0.625), ylabel='CIE v\'')

    return fig, ax

'''
    Below here are color space conversions ported from colour to make them numpy
    ufuncs supporting array vectorization.  For more information, see colour:
    https://github.com/colour-science/colour/
'''

def XYZ_to_xyY(XYZ):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with first dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

            `numpy.ndarray`: Y coordinates.

    '''
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    Y = Y
    shape = x.shape
    return np.stack((x, y, Y), axis=len(shape))

def XYZ_to_xy(XYZ):
    ''' Converts XYZ points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with first dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    xyY = XYZ_to_xyY(XYZ)
    return xyY_to_xy(xyY)

def XYZ_to_uv(XYZ):
    ''' Converts XYZ points to uv points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with first dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: u coordinates.

            `numpy.ndarray`: u coordinates.

    '''
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    u = (4 * X) / (X + 15 * Y + 3 * Z)
    v = (9 * Y) / (X + 15 * Y + 3 * Z)

    shape = u.shape
    return np.stack((u, v), axis=len(shape))

def xyY_to_xy(xyY):
    ''' converts xyY points to xy points.

    Args:
        xyY (`numpy.ndarray`): ndarray with first dimension corresponding to
            x, y, Y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    shape = xyY.shape
    if shape[-1] is 2:
        return xyY
    else:
        x, y, Y = xyY

        shape = x.shape
        return np.stack((x, y), axis=len(shape))

def xyY_to_XYZ(xyY):
    ''' converts xyY points to XYZ points.

    Args:
        xyY (`numpy.ndarray`): ndarray with first dimension corresponding to
            x, y, Y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            `numpy.ndarray`: Z coordinates.

    '''
    x, y, Y = xyY[..., 0], xyY[..., 1], xyY[..., 2]
    X = (x * Y) / y
    Y = Y
    Z = ((1 - x - y) * Y) / y

    shape = X.shape
    return np.stack((X, Y, Z), axis=len(shape))

def xy_to_xyY(xy, Y=1):
    ''' converts xy points to xyY points.

    Args:
        xy (`numpy.ndarray`): ndarray with first dimension corresponding to
            x, y.

        Y (`numpy.ndarray`): Y value to fill with.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

            `numpy.ndarray`: Y coordinates.

    '''
    shape = xy.shape
    if shape[-1] is 3:
        return xy
    else:
        x, y = xy[..., 0], xy[..., 1]
        Y = np.ones(x.shape) * Y

        shape = x.shape
        return np.stack((x, y, Y), axis=len(shape))

def xy_to_XYZ(xy):
    ''' converts xy points to xyY points.

    Args:
        xy (`numpy.ndarray`): ndarray with first dimension corresponding to
            x, y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            `numpy.ndarray`: Z coordinates.

    '''
    xyY = xy_to_xyY(xy)
    return xyY_to_XYZ(xyY)

def Luv_to_XYZ(Luv):
    ''' Converts Luv points to XYZ points.

    Args:
        Luv (`numpy.ndarray`): ndarray with first dimension corresponding to
            L, u, v.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            'numpy.ndarray`: Z coordinates.

    '''
    L, u, v = Luv[..., 0], Luv[..., 1], Luv[..., 2]
    XYZ_D50 = [0.9642, 1.0000, 0.8251]
    X_r, Y_r, Z_r = XYZ_D50 # tsplit(xyY_to_XYZ(xy_to_xyY(illuminant)))

    Y = np.where(L > CIE_E * CIE_K, ((L + 16) / 116) ** 3, L / CIE_K)

    a = 1 / 3 * ((52 * L / (u + 13 * L * (4 * X_r /
                                          (X_r + 15 * Y_r + 3 * Z_r)))) - 1)
    b = -5 * Y
    c = -1 / 3.0
    d = Y * (39 * L / (v + 13 * L * (9 * Y_r /
                                     (X_r + 15 * Y_r + 3 * Z_r))) - 5)

    X = (d - b) / (a - c)
    Z = X * a + b

    shape = X.shape
    return np.stack((X, Y, Z), axis=len(shape))

def Luv_to_uv(Luv):
    ''' Converts Luv points to uv points.

    Args:
        Luv (`numpy.ndarray`): ndarray with first dimension corresponding to
            L, u, v.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: u coordinates.

            `numpy.ndarray`: v coordinates.

    '''
    XYZ = Luv_to_XYZ(Luv)
    return XYZ_to_uv(XYZ)

def Luv_uv_to_xy(uv):
    ''' Converts Luv u,v points to xyY x,y points.

    Args:
        uv (`numpy.ndarray`): ndarray with first dimension corresponding to
            u, v.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    u, v = uv[..., 0], uv[..., 1]
    x = (9 * u) / (6 * u - 16 * v + 12)
    y = (4 * v) / (6 * u - 16 * v + 12)

    shape = x.shape
    return np.stack((x, y), axis=len(shape))

def XYZ_to_AdobeRGB(XYZ, illuminant='D65'):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with first dimension corresponding to
            X, Y, Z.

        illuminant (`str`): which illuminant to use, either D65 or D50.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: R coordinates.

            `numpy.ndarray`: G coordinates.

            `numpy.ndarray`: B coordinates.

    '''
    if illuminant.upper() == 'D65':
        invmat = COLOR_MATRICIES['AdobeRGB']['D65']
    elif illuminant.upper() == 'D50':
        invmat = COLOR_MATRICIES['AdobeRGB']['D50']
    else:
        raise ValueError('Must use D65 or D50 illuminant.')

    return XYZ_to_RGB(XYZ, invmat)

def XYZ_to_sRGB(XYZ, illuminant='D65'):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with first dimension corresponding to
            X, Y, Z.

        illuminant (`str`): which illuminant to use, either D65 or D50.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: R coordinates.

            `numpy.ndarray`: G coordinates.

            `numpy.ndarray`: B coordinates.

    Notes:
        Returns are linear, need to be raised to 2.4 power to make correct
            values for viewing.

    '''
    if illuminant.upper() == 'D65':
        invmat = COLOR_MATRICIES['sRGB']['D65']
    elif illuminant.upper() == 'D50':
        invmat = COLOR_MATRICIES['sRGB']['D50']
    else:
        raise ValueError('Must use D65 or D50 illuminant.')

    return XYZ_to_RGB(XYZ, invmat)

def XYZ_to_RGB(XYZ, conversion_matrix):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with first dimension corresponding to
            X, Y, Z.

        conversion_matrix (`str`): conversion matrix to use to convert XYZ
            to RGB values.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: R coordinates.

            `numpy.ndarray`: G coordinates.

            `numpy.ndarray`: B coordinates.

    '''
    if len(XYZ.shape) == 1:
        return np.matmul(conversion_matrix, XYZ)
    else:
        return np.tensordot(XYZ, conversion_matrix, axes=((2), (1)))
