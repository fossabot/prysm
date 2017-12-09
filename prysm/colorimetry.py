''' optional tools for colorimetry, wrapping the color-science library, see:
    http://colour-science.org/
'''
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
from scipy.constants import c, h, k
from scipy.interpolate import interp1d
# c - speed of light
# h - planck constant
# k - boltzman constant

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

try:
    import colour
except ImportError:
    # Spectrum objects can be used without colour
    pass

from prysm.conf import config
from prysm.util import share_fig_ax, correct_gamma
from prysm.mathops import atan2, pi, cos, sin, exp, sqrt

# some CIE constants
CIE_K = 24389 / 27
CIE_E = 216 / 24389

# color matching functions will be populated as needed
color_matching_functions = {}

# CCT values in L*u*v* coordinates, see the following for the source of these values:
# https://www.osapublishing.org/josa/abstract.cfm?uri=josa-58-11-1528
cct_K = np.asarray([
    1667, 1739, 1818, 1905,
    2000, 2105, 2222, 2353, 2500, 2667, 2857,
    3077, 3333, 3636,
    4000, 4444,
    5000, 5714,
    6667, 8000, 10000,
    11111, 12500, 14286, 16667,
    20000, 25000, 33333,
    50000, 100000, 1e20,
], dtype=config.precision)

cct_urd = np.asarray([
    600, 575, 550, 525,
    500, 475, 450, 425, 400, 375, 350,
    325, 300, 275,
    250, 225,
    200, 175,
    150, 125, 100,
    90, 80, 70, 60,
    50, 40, 30,
    20, 10, 0,
], dtype=config.precision)

cct_u = np.asarray([
    0.33713, 0.32920, 0.32119, 0.31310,
    0.30496, 0.29676, 0.28854, 0.28032, 0.27210, 0.26394, 0.25585,
    0.24787, 0.24005, 0.23243,
    0.22507, 0.21804,
    0.21140, 0.20523,
    0.19960, 0.19461, 0.19031,
    0.1879, 0.18739, 0.18611, 0.18494,
    0.18388, 0.18293, 0.18208,
    0.18132, 0.18065, 0.18006
], dtype=config.precision)

cct_v = np.asarray([
    0.36051, 0.36038, 0.36011, 0.35968,
    0.35906, 0.35822, 0.35713, 0.35575, 0.35405, 0.35198, 0.34948,
    0.34653, 0.34305, 0.33901,
    0.33436, 0.32906,
    0.32309, 0.31645,
    0.30918, 0.30139, 0.29325,
    0.28995, 0.28666, 0.28340, 0.28020,
    0.27708, 0.27407, 0.27118,
    0.26845, 0.26589, 0.26352
], dtype=config.precision)

cct_dvdu = np.asarray([
    -113.8, -40.41, -23.20, -15.56,
    -11.29, -8.572, -6.711, -5.365, -4.355, -3.576, -2.960,
    -2.465, -2.061, -1.728,
    -1.450, -1.216,
    -1.017, -0.8484,
    -0.7043, -0.5817, -0.4787,
    -0.4426, -0.4094, -0.3790, -0.3515,
    -0.3267, -0.3047, -0.2854,
    -0.2687, -0.2548, -0.2434,
], dtype=config.precision)

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


def prepare_cie_1931_2deg_observer():
    ''' Prepares the CIE 1931 standard 2 degree observer, if it is not already
        cached.
    '''
    tmp_list = []
    p = Path(__file__) / './cie_xyz_1931_tristimulus_5nm.csv'
    with open(p, 'r') as fid:
        reader = csv.reader(fid)
        for row in reader:
            tmp_list.append(row)

    values = np.asarray(tmp_list, dtype=config.precision)
    wvl, X, Y, Z = values[:, 0], values[:, 1], values[:, 2], values[:, 3]
    color_matching_functions['cie1931_2deg'] = {
        'wvl': wvl,
        'X': X,
        'Y': Y,
        'Z': Z
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


def blackbody_spectral_power_distribution(temperature, wavelengths):
    ''' Computes the spectral power distribution of a black body at a given
        temperature.

    Args:
        temperature (`float`): body temp, in Kelvin.

        wavelengths (`numpy.ndarray`): array of wavelengths, in nanometers.

    Returns:
        numpy.ndarray: spectral power distribution in units of W/m^2/nm

    '''
    wavelengths = wavelengths.astype(config.precision) / 1e9
    return (2 * h * c ** 2) / (wavelengths ** 5) * \
        1 / (exp((h * c) / (wavelengths * k * temperature) - 1))


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
    return Spectrum(wvl, vals / vis_values_max)


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
    if 'colour' not in globals():  # or in locals
            raise ImportError('prysm colorimetry requires the colour package, '
                              'see http://colour-science.org/installation-guide/')


def cie_1976_plot(xlim=(-0.09, 0.68), ylim=None, samples=200, fig=None, ax=None):
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
    # yet another set for annotation.
    wvl_line = np.arange(400, 700, 2)
    wvl_line_uv = XYZ_to_uvprime(colour.wavelength_to_XYZ(wvl_line))

    wvl_annotate = [360, 400, 455, 470, 480, 490,
                    500, 510, 520, 540, 555, 570, 580, 590,
                    600, 615, 630, 700, 830]

    wvl_mask = [400, 430, 460, 465, 470, 475, 480, 485, 490, 495,
                500, 505, 510, 515, 520, 525, 530, 535, 570, 700]

    wvl_mask_uv = XYZ_to_uvprime(colour.wavelength_to_XYZ(wvl_mask))

    # make equally spaced u,v coordinates on a grid
    u = np.linspace(xlim[0], xlim[1], samples)
    v = np.linspace(ylim[0], ylim[1], samples)
    uu, vv = np.meshgrid(u, v)

    # set values outside the horseshoe to a safe value that won't blow up

    # stack u and v for vectorized computations
    uuvv = np.stack((vv, uu), axis=2)

    triangles = Delaunay(wvl_mask_uv, qhull_options='QJ Qf')
    wvl_mask = triangles.find_simplex(uuvv) < 0

    xy = uvprime_to_xy(uuvv)
    xyz = xy_to_XYZ(xy)
    dat = XYZ_to_sRGB(xyz)
    # normalize and clip
    dat /= np.max(dat, axis=2)[..., np.newaxis]
    dat = np.clip(dat, 0, 1)

    # now make an alpha/transparency mask to hide the background
    # and flip u,v axes because of column-major symantics
    alpha = np.ones((samples, samples))
    alpha[wvl_mask] = 0
    dat = np.swapaxes(np.dstack((dat, alpha)), 0, 1)

    # lastly, duplicate the lowest wavelength so that the boundary line is closed
    wvl_line_uv = np.vstack((wvl_line_uv, wvl_line_uv[0, :]))

    fig, ax = share_fig_ax(fig, ax)
    ax.imshow(dat,
              extent=[*xlim, *ylim],
              interpolation='bilinear',
              origin='lower')
    ax.plot(wvl_line_uv[:, 0], wvl_line_uv[:, 1], ls='-', c='0.25', lw=2)
    fig, ax = cie_1976_wavelength_annotations(wvl_annotate, fig=fig, ax=ax)
    ax.set(xlim=xlim, xlabel='CIE u\'',
           ylim=ylim, ylabel='CIE v\'')

    return fig, ax


def cie_1976_wavelength_annotations(wavelengths, fig=None, ax=None):
    ''' Draws lines normal to the spectral locust on a CIE 1976 diagram and
        writes the text for each wavelength.

    Args:
        wavelengths (`iterable`): set of wavelengths to annotate.

        fig (`matplotlib.figure.Figure`): figure to draw on.

        ax (`matplotlib.axes.Axis`): axis to draw in.

    Returns:

        `tuple` containing:

            `matplotlib.figure.Figure`: figure containing the annotations.

            `matplotlib.axes.Axis`: axis containing the annotations.

    Notes:
        see SE:
        https://stackoverflow.com/questions/26768934/annotation-along-a-curve-in-matplotlib

    '''
    # some tick parameters
    tick_length = 0.025
    text_offset = 0.06

    # convert wavelength to u' v' coordinates
    wavelengths = np.asarray(wavelengths)
    idx = np.arange(1, len(wavelengths) - 1, dtype=int)
    wvl_lbl = wavelengths[idx]
    uv = XYZ_to_uvprime(colour.wavelength_to_XYZ(wavelengths))
    u, v = uv[..., 0][idx], uv[..., 1][idx]
    u_last, v_last = uv[..., 0][idx - 1], uv[..., 1][idx - 1]
    u_next, v_next = uv[..., 0][idx + 1], uv[..., 1][idx + 1]

    angle = atan2(v_next - v_last, u_next - u_last) + pi / 2
    cos_ang, sin_ang = cos(angle), sin(angle)
    u1, v1 = u + tick_length * cos_ang, v + tick_length * sin_ang
    u2, v2 = u + text_offset * cos_ang, v + text_offset * sin_ang

    fig, ax = share_fig_ax(fig, ax)
    tick_lines = LineCollection(np.c_[u, v, u1, v1].reshape(-1, 2, 2), color='0.25', lw=1.25)
    ax.add_collection(tick_lines)
    for i in range(len(idx)):
        ax.text(u2[i], v2[i], str(wvl_lbl[i]), va="center", ha="center", clip_on=True)

    return fig, ax


def cie_1976_plankian_locust(trange=(2000, 10000), num_points=100,
                             isotemperature_lines_at=None, isotemperature_du=0.025,
                             fig=None, ax=None):
    ''' draws the plankian locust on the CIE 1976 color diagram.

    Args:
        trange (`iterable`): (min,max) color temperatures.

        num_points (`int`): number of points to compute.

        isotemperature_lines_at (`iterable`): CCTs to plot isotemperature lines at,
            defaults to [2000, 3000, 4000, 5000, 6500, 10000] if None.
            set to False to not plot lines.

        isotemperature_du (`float`): delta-u, parameter, length in x of the isotemperature lines.

        fig (`matplotlib.figure.Figure`): figure to plot in.

        ax (`matplotlib.axes.Axis`): axis to plot in.

    Returns:
        `tuple` containing:

            `matplotlib.figure.Figure`. figure containing the plot.

            `matplotlib.axes.Axis`: axis containing the plot.

    '''

    # compute the u', v' coordinates of the temperatures
    temps = np.linspace(trange[0], trange[1], num_points)
    interpf_u = interp1d(cct_K, cct_u)
    interpf_v = interp1d(cct_K, cct_v)
    u = interpf_u(temps)
    v = interpf_v(temps) * 1.5  # x1.5 converts 1960 uv to 1976 u' v'

    # if plotting isotemperature lines, compute the upper and lower points of
    # each line and connect them.
    plot_isotemp = True
    if isotemperature_lines_at is None:
        isotemperature_lines_at = np.asarray([2000, 3000, 4000, 5000, 6500, 10000])
        u_iso = interpf_u(isotemperature_lines_at)
        v_iso = interpf_v(isotemperature_lines_at)
        interpf_dvdu = interp1d(cct_u, cct_dvdu)

        dvdu = interpf_dvdu(u_iso)
        du = isotemperature_du / dvdu

        u_high = u_iso + du / 2
        u_low = u_iso - du / 2
        v_high = (v_iso + du / 2 * dvdu) * 1.5  # factors of 1.5 convert from uv to u'v'
        v_low = (v_iso - du / 2 * dvdu) * 1.5
    elif isotemperature_lines_at is False:
        plot_isotemp = False

    fig, ax = share_fig_ax(fig, ax)
    ax.plot(u, v, c='0.15')
    if plot_isotemp is True:
        for ul, uh, vl, vh in zip(u_low, u_high, v_low, v_high):
            ax.plot([ul, uh], [vl, vh], c='0.15')

    return fig, ax

'''
    Below here are color space conversions ported from colour to make them numpy
    ufuncs supporting array vectorization.  For more information, see colour:
    https://github.com/colour-science/colour/
'''


def safely_get_cmf(observer='1931_2deg'):
    ''' Safely returns the color matching function dictionary for the specified
        observer.

    Args:
        observer (`str): the observer to return.

    Returns:
        `dict`: cmf dict.

    '''
    try:
        cmf = color_matching_functions[observer]
    except KeyError as e:
        prepare_cie_1931_2deg_observer()
        cmf = color_matching_functions[observer]

    return cmf


def spectrum_to_XYZ_emissive(wvl, values, cmf='1931_2deg'):
    ''' Converts a reflective or transmissive spectrum to XYZ coordinates.

    Args:
        wvl (`numpy.ndarray`): wavelengths the data is sampled at, in nm.

        values (`numpy.ndarray`): values of the spectrum at each wvl sample.

        cmf (`str`): which color matching function to use, defaults to
            CIE 1931 2 degree observer.

    Returns:
        `tuple` containing:

            `float`: X

            `float`: Y

            `float`: Z

    '''
    if cmf.lower() is not '1931_2deg':
        raise ValueError('Must use 1931 2 degree standard observer (cmf=1931_2deg)')

    cmf = safely_get_cmf(cmf)

    wvl_cmf = cmf.wvl
    if not np.allclose(wvl_cmf, wvl):
        dat_interpf = interp1d(wvl, values, kind='linear', fill_value=0, assume_sorted=True)
        values = dat_interpf(wvl_cmf)

    X = k * np.trapz(values * cmf.X)
    Y = k * np.trapz(values * cmf.Y)
    Z = k * np.traps(values * cmf.Z)
    return (X, Y, Z)


def spectrum_to_XYZ_nonemissive(wvl, values, illuminant='bb_6500', cmf='1931_2deg'):
    ''' Converts a reflective or transmissive spectrum to XYZ coordinates.

    Args:
        wvl (`numpy.ndarray`): wavelengths the data is sampled at, in nm.

        values (`numpy.ndarray`): values of the spectrum at each wvl sample.

        illuminant (`str`): reference illuminant, of the form "bb_temperature".
            TODO: add D65, D50, etc.

        cmf (`str`): which color matching function to use, defaults to
            CIE 1931 2 degree observer.

    Returns:
        `tuple` containing:

            `float`: X

            `float`: Y

            `float`: Z

    '''
    if cmf.lower() is not '1931_2deg':
        raise ValueError('Must use 1931 2 degree standard observer (cmf=1931_2deg)')

    cmf = safely_get_cmf(cmf)

    try:
        if illuminant[2] == '_':
            # black body
            _, temperature = illuminant.split('_')
            ill_type = 'blackbody'
        else:
            raise ValueError('not blackbody')
    except (ValueError, IndexError) as err:
        # standard illuminant, not implemented
        raise ValueError('Must use black body illuminants')

    wvl_cmf = cmf.wvl
    if not np.allclose(wvl_cmf, wvl):
        dat_interpf = interp1d(wvl, values, kind='linear', fill_value=0, assume_sorted=True)
        values = dat_interpf(wvl_cmf)

    if ill_type is 'blackbody':
        ill_spectrum = blackbody_spectral_power_distribution(temperature, cmf.wvl)
    else:
        ill_spectrum = np.zeros(cmf.wvl.shape)

    k = 100 / np.trapz(ill_spectrum)
    X = k * np.trapz(values * ill_spectrum * cmf.X)
    Y = k * np.trapz(values * ill_spectrum * cmf.Y)
    Z = k * np.traps(values * ill_spectrum * cmf.Z)
    return (X, Y, Z)


def make_cieluv_isotemperature_line_points(temp, length=0.025):
    ''' Computes the upper and lower (u',v') points for a given color
        temperature.

    Args:
        temp (`float`): temperature to make isotemperature points for

        length (`float`): total length of the isotemperature line.

    Returns:
        `numpy.ndarray` with last dim (u',v')

    '''
    cmf = safely_get_cmf()
    spectrum = blackbody_spectral_power_distribution(temp, cmf.wvl)
    xyz = spectrum_to_XYZ_nonemissive(cmf.wvl, spectrum, illuminant=f'bb_{temp}')
    uv = XYZ_to_uvprime(xyz)

    pass


def XYZ_to_xyY(XYZ, assume_nozeros=True, zero_ref_illuminant='D65'):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

        assume_nozeros (`bool`): assume there are no zeros present, computation
            will run faster as `True`, if `False` will correct for all-zero
            values.

        zero_ref_illuminant (`str): string for reference illuminant in the case
            where X==Y==Z==0.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

            `numpy.ndarray`: Y coordinates.

    Notes:
        zero_ref_illuminant is unimplemented, forced to D65 at the time being.

    '''
    x_D65, y_D65 = 0.3128, 0.3290

    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]

    if not assume_nozeros:
        zero_X = np.where(X == 0)
        zero_Y = np.where(Y == 0)
        zero_Z = np.where(Z == 0)
        allzeros = np.all(np.dstack((zero_X, zero_Y, zero_Z)))
        X[allzeros] = 0.3
        Y[allzeros] = 0.3
        Z[allzeros] = 0.3

    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    Y = Y
    shape = x.shape

    if not assume_nozeros:
        x[allzeros] = x_D65
        y[allzeros] = y_D65

    return np.stack((x, y, Y), axis=len(shape))


def XYZ_to_xy(XYZ):
    ''' Converts XYZ points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: x coordinates.

            `numpy.ndarray`: y coordinates.

    '''
    xyY = XYZ_to_xyY(XYZ)
    return xyY_to_xy(xyY)


def XYZ_to_uvprime(XYZ):
    ''' Converts XYZ points to u'v' points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
            X, Y, Z.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: u' coordinates.

            `numpy.ndarray`: u' coordinates.

    '''
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    u = (4 * X) / (X + 15 * Y + 3 * Z)
    v = (9 * Y) / (X + 15 * Y + 3 * Z)

    shape = u.shape
    return np.stack((u, v), axis=len(shape))


def XYZ_to_Luv(xyz, refwhite='bb_6500'):
    ''' Converts XYZ to Luv coordinates with a given white point.

    Args:
        xyz (`numpy.ndarray`): array with last dimension X, Y, Z.

        refwhite (`str`): reference white, must be a 6500K black body.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: L coordinates.

            `numpy.ndarray`: u coordinates.

            `numpy.ndarray`: v coordinates.

    '''
    xyz = np.asarray(xyz)
    upvp = XYZ_to_uvprime(xyz)
    up, vp = upvp[..., 0], upvp[..., 1]

    wvl = np.arange(360, 830, 5)
    spectrum = blackbody_spectral_power_distribution(6500, wvl)
    ref_xyz = spectrum_to_XYZ_emissive(wvl, spectrum)
    ref_upvp = XYZ_to_uvprime(ref_xyz)
    ref_up, ref_vp = ref_upvp[..., 0], ref_upvp[..., 1]

    Y = xyz[..., 1]
    yr = Y / ref_xyz[1]

    L_case_one = np.where(yr > CIE_E)
    L = CIE_K * yr
    L[L_case_one] = 116 * yr ** (1 / 3) - 16
    u = 13 * L * (up - ref_up)
    v = 13 * L * (vp - ref_vp)

    shape = xyz.shape
    return np.stack((L, u, v), axis=len(shape))


def xyY_to_xy(xyY):
    ''' converts xyY points to xy points.

    Args:
        xyY (`numpy.ndarray`): ndarray with last dimension corresponding to
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
        xyY (`numpy.ndarray`): ndarray with last dimension corresponding to
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
        xy (`numpy.ndarray`): ndarray with last dimension corresponding to
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
        xy (`numpy.ndarray`): ndarray with last dimension corresponding to
            x, y.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            `numpy.ndarray`: Z coordinates.

    '''
    xyY = xy_to_xyY(xy)
    return xyY_to_XYZ(xyY)


def xy_to_CCT(xy):
    ''' Computes the correlated color temperature given x,y chromaticity coordinates.

    Args:
        xy (`iterable`): x, y chromaticity coordinates.

    Returns:
        `float`: CCT.

    '''
    xy = np.asarray(xy)
    x, y = xy[..., 0], xy[..., 1]
    n = (x - 0.3320) / (0.1858 - y)
    return 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.3


def Luv_to_XYZ(Luv):
    ''' Converts Luv points to XYZ points.

    Args:
        Luv (`numpy.ndarray`): ndarray with last dimension corresponding to
            L, u, v.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: X coordinates.

            `numpy.ndarray`: Y coordinates.

            'numpy.ndarray`: Z coordinates.

    '''
    L, u, v = Luv[..., 0], Luv[..., 1], Luv[..., 2]
    XYZ_D50 = [0.9642, 1.0000, 0.8251]
    X_r, Y_r, Z_r = XYZ_D50

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


def Luv_to_uvprime(Luv):
    ''' Converts Luv points to u'v' points.

    Args:
        Luv (`numpy.ndarray`): ndarray with last dimension corresponding to
            L*, u*, v*.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: u coordinates.

            `numpy.ndarray`: v coordinates.

    '''
    XYZ = Luv_to_XYZ(Luv)
    return XYZ_to_uvprime(XYZ)


def Luv_to_chroma_hue(luv):
    ''' Converts L*u*v* coordiantes to a chroma and hue.

    Args:
        luv (`numpy.ndarray`): array with last dimension L*, u*, v*.

    Returns:
        `numpy.ndarray` with last dimension corresponding to C* and h.

    '''
    luv = np.asarray(luv)
    u, v = luv[..., 1], luv[..., 2]
    C = sqrt(u**2 + v**2)
    h = atan2(v, u)

    shape = luv.shape
    return np.stack((C, h), axis=len(shape))


def uvprime_to_xy(uv):
    ''' Converts u' v' points to xyY x,y points.

    Args:
        uv (`numpy.ndarray`): ndarray with last dimension corresponding to
            u', v'.

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


def uvprime_to_Luv(uv):
    ''' Converts u' v' points to xyY x,y points.

    Args:
        uv (`numpy.ndarray`): ndarray with last dimension corresponding to
            u', v'.

    Returns:
        `tuple` containing:

            `numpy.ndarray`: L coordinates.

            `numpy.ndarray`: u coordinates.

            `numpy.ndarray`: v coordinates.

    '''
    xy = uvprime_to_xy(uv)
    xyz = xy_to_XYZ(xy)
    luv = XYZ_to_Luv(xyz)
    return luv


def XYZ_to_AdobeRGB(XYZ, illuminant='D65'):
    ''' Converts xyz points to xy points.

    Args:
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
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
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
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
        XYZ (`numpy.ndarray`): ndarray with last dimension corresponding to
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
