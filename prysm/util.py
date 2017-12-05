''' Utility functions
'''
import numpy as np
from matplotlib import pyplot as plt

from operator import itemgetter

def is_odd(int):
    ''' Determines if an interger is odd using binary operations.

    Args:
        int (`int`): an integer.

    Returns:
        bool: true if odd, False if even.

    '''
    return int & 0x1

def is_power_of_2(value):
    ''' Checks if a value is a power of 2 using binary operations.

    Args:
        value (`number`): value to check.

    Returns:
        bool: true if the value is a power of two, False if the value is no.

    Notes:
        c++ inspired implementation, see SO:
        https://stackoverflow.com/questions/29480680/finding-if-a-number-is-a-power-of-2-using-recursion

    '''
    return bool(value and not value&(value-1))


def pupil_sample_to_psf_sample(pupil_sample, num_samples, wavelength, efl):
    ''' Converts pupil sample spacing to PSF sample spacing.

    Args:
        pupil_sample (`float`): sample spacing in the pupil plane.

        num_samples (`int`): number of samples present in both planes (must be equal).

        wavelength (`float`): wavelength of light, in microns.

        efl (`float`): effective focal length of the optical system in mm.

    Returns:
        float: the sample spacing in the PSF plane.

    '''
    return (wavelength * efl * 1e3) / (pupil_sample * num_samples)

def psf_sample_to_pupil_sample(psf_sample, num_samples, wavelength, efl):
    ''' Converts PSF sample spacing to pupil sample spacing.

    Args:
        psf_sample (`float`): sample spacing in the PSF plane.

        num_samples (`int`): number of samples present in both planes (must be equal).

        wavelength (`float`): wavelength of light, in microns.

        efl (`float`): effective focal length of the optical system in mm.

    Returns:
        float: the sample spacing in the pupil plane.

    '''
    return (psf_sample * num_samples) / (wavelength * efl * 1e3)

def correct_gamma(img, encoding=2.2):
    ''' Applies an inverse gamma curve to image data that linearizes the given encoding.

    Args:
        img (`numpy.ndarray`): array of image data, floats avoid quantization error.

        encoding (`float`): gamma the data is encoded in (1.0 is linear).

    Returns:
        numpy.ndarray:  Array of corrected data.

    '''
    return np.power(img, (1/float(encoding)))

def fold_array(array, axis=1):
    ''' Folds an array in half over the given axis and averages.

    Args:
        array (`numpy.ndarray`): 2d array to fold.

        axis (`int`): axis to fold over.

    Returns
        numpy.ndarray:  new array.

    '''

    xs, ys = array.shape
    if axis is 1:
        xh = int(np.floor(xs/2))
        left_chunk = array[:, :xh]
        right_chunk = array[:, xh:]
        folded_array = np.concatenate((right_chunk[:, :, np.newaxis],
                                       np.flip(np.flip(left_chunk, axis=1),
                                               axis=0)[:, :, np.newaxis]),
                                      axis=2)
    else:
        yh = int(np.floor(ys/2))
        top_chunk = array[:yh, :]
        bottom_chunk = array[yh:, :]
        folded_array = np.concatonate((bottom_chunk[:, :, np.newaxis],
                                       np.flip(np.flip(top_chunk, axis=1),
                                               axis=0)[:, :, np.newaxis]),
                                      axis=2)
    return np.average(folded_array, axis=2)

def share_fig_ax(fig=None, ax=None, numax=1, sharex=False, sharey=False):
    ''' Reurns the given figure and/or axis if given one.  If they are None, creates a new fig/ax

    Args:
        fig (`pyplot.figure`): figure.

        ax (`pyplot.axis`): axis or array of axes.

        numax (`int`): number of axes in the desired figure.
                     1 for most plots, 3 for plot_fourier_chain.

    Returns:
        pyplot.figure:  A figure object.

        pyplot.axis:  An axis object.

    '''
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=numax, sharex=sharex, sharey=sharey)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    return fig, ax

def rms(array):
    ''' Returns the RMS value of the valid elements of an array

    Args:
        array (`numpy.ndarray`)

    Returns:
        float.  RMS of the array

    '''
    non_nan = np.isfinite(array)
    return np.sqrt(np.mean(np.square(array[non_nan])))

def guarantee_array(variable):
    ''' Guarantees that a varaible is a numpy ndarray and supports -, *, +, and other operators

    Args:
        variable (`number` or `numpy.ndarray`): variable to coalesce

    Returns:
        (type).  Which supports * / and other operations with arrays

    '''
    if type(variable) in [float, np.ndarray, np.int32, np.int64, np.float32, np.float64]:
        return variable
    elif type(variable) is int:
        return float(variable)
    elif type(variable) is list:
        return np.asarray(variable)
    else:
        raise ValueError(f'variable is of invalid type {type(variable)}')

def ecdf(x):
    ''' Computes the empirical cumulative distribution function of a dataset

    Args:
        x (`iterable`): Data.

    Returns:
        tuple containing:
            `numpy.ndarray`: sorted data.

            `numpy.ndarray`: cumulative distribution function of the data.

    '''
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys

def sort_xy(x, y):
    ''' Sorts a pair of x and y iterables, returning arrays in order of
        ascending x.

    Args:
        x (`iterable`): a list, numpy ndarray, or other iterable to sort by.

        y (`iterable`): a list, numpy ndarray, or other iterable that is y=f(x).

    Returns:
        tuple containing:
            `iterable`: an iterable containing the sorted x elements.

            `iterable`: an iterable containing the sorted y elements.

    '''
    # zip x and y, sort by the 0th element (x) of each tuple in zip()
    _ = sorted(zip(x,y), key=itemgetter(0))
    sorted_x, sorted_y = zip(*_)
    return sorted_x, sorted_y

def smooth(x, window_len=3, window='flat'):
    ''' Smooth data.

    adapted from scipy signal smoothing cookbook,
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Args:
        x (`iterable`): the input signal.

        window_len: the dimension of the smoothing window; should be an odd integer.

        window: the type of window from ('flat', 'hanning', 'hamming',
            'bartlett', 'blackman').  Use flat for moving average.

    Returns:
        `iterable`: The smoothed signal.

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    '''

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('Data must be 1D.')

    if x.size < window_len:
        raise ValueError('Data must be larger than window.')

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError('Window must be one of flat, hanning, hamming, bartlett, blackman')

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window.lower() == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(int(np.floor(window_len / 2)) - 1):-(int(np.ceil(window_len / 2)))]
