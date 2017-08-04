import numpy as np
from matplotlib import pyplot as plt

def pupil_sample_to_psf_sample(pupil_sample, num_samples, wavelength, efl):
    '''Converts pupil sample spacing to PSF sample spacing

    Args:
        pupil_sample (float): sample spacing in the pupil plane
        num_samples (int): number of samples present in both planes (must be equal)
        wavelength (float): wavelength of light, in microns
        efl (float): effective focal length of the optical system in mm

    Returns:
        float.  The sample spacing in the PSF plane.
    '''
    return (wavelength * efl * 1e3) / (pupil_sample * num_samples)

def psf_sample_to_pupil_sample(psf_sample, num_samples, wavelength, efl):
    '''Converts PSF sample spacing to pupil sample spacing

    Args:
        psf_sample (float): sample spacing in the PSF plane
        num_samples (int): number of samples present in both planes (must be equal)
        wavelength (float): wavelength of light, in microns
        efl (float): effective focal length of the optical system in mm

    Returns:
        float.  The sample spacing in the pupil plane.
    '''
    return (psf_sample * num_samples) / (wavelength * efl * 1e3)

def correct_gamma(img, encoding=2.2):
    '''Applies an inverse gamma curve to image data that linearizes the given encoding

    Args:
        img (numpy.array): array of image data, floats avoid quantization error
        encoding (float): gamma the data is encoded in (1.0 is linear)

    Returns:
        numpy.array.  Array of corrected data.
    '''
    return np.power(img, (1/float(encoding)))

def fold_array(array):
    '''folds an array in half over the given axis and averages

    Args:
        array (numpy.array): 2d array to fold

    Returns
        numpy.array.  new array
    '''
    xs, ys = array.shape
    xh = int(np.floor(xs/2))
    left_chunk = array[:, :xh]
    right_chunk = array[:, xh:]
    folded_array = np.concatenate((right_chunk[:, :, np.newaxis],
                                   np.flip(np.flip(left_chunk, axis=1), axis=0)[:, :, np.newaxis]),
                                  axis=2)
    return np.average(folded_array, axis=2)

def share_fig_ax(fig=None, ax=None, numax=1):
    '''Reurns the given figure and/or axis if given one.  If they are None, creates a new fig/ax

    Args:
        fig (pyplot.figure): figure
        ax (pyplot.axis): axis or array of axes
        numax (int): number of axes in the desired figure.  1 for most plots, 3 for plot_fourier_chain

    Returns:
        pyplot.figure.  A figure object.
        pyplot.axis.  An axis object
    '''
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=numax, dpi=100)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    return fig, ax

def rms(array):
    '''Returns the RMS value of an array

    Args:
        array (numpy.ndarray)

    Returns:
        rms value
    '''
    non_nan = np.isfinite(array)
    return np.sqrt(np.mean(np.square(array[non_nan])))

def guarantee_array(variable):
    '''Guarantees that a varaible is a numpy ndarray and supports -, *, +, and other operators

    Args:
        variable (float or numpy.ndarray): variable to coalesce

    Returns:
        numpy ndarray equivalent
    '''
    if type(variable) in [float, np.ndarray, np.int32, np.int64, np.float32, np.float64]:
        return variable
    elif type(variable) is int:
        return float(variable)
    elif type(variable) is list:
        return np.asarray(variable)
    else:
        raise ValueError(f'variable is of invalid type {type(variable)}')