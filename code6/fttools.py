'''
Supplimental tools for computing fourier transforms
'''
import numpy as np

def pad2d(array, factor=1):
    '''
        pads a 2D array such that the output array is 2*factor times larger in each dimmension than
        the input array
    '''
    x, y = array.shape
    pad_shape = ((x*factor, x*factor), (y*factor, y*factor))
    return np.pad(array, pad_width=pad_shape, mode='constant', constant_values=0)

def forward_ft_unit(sample_spacing, samples):
    f_s = int(floor(samples / 2))
    return 1 / (sample_spacing / 1e3) * np.arange(-f_s, f_s) / samples
def matrix_dft(f, alpha, npix, shift=None, unitary=False):
    '''
    A technique shamelessly stolen from Andy Kee @ NASA JPL
    Is it magic or math?
    '''
    if np.isscalar(alpha):
        ax = ay = alpha
    else:
        ax = ay = np.asarray(alpha)

    f = np.asarray(f)
    m, n = f.shape

    if np.isscalar(npix):
        M = N = npix
    else:
        M = N = np.asarray(npix)

    if shift is None:
        sx = sy = 0
    else:
        sx = sy = np.asarray(shift)

    # Y and X are (r,c) coordinates in the (m x n) input plane, f
    # V and U are (r,c) coordinates in the (M x N) output plane, F
    X = np.arange(n) - np.floor(n/2) - sx
    Y = np.arange(m) - np.floor(m/2) - sy
    U = np.arange(N) - np.floor(N/2) - sx
    V = np.arange(M) - np.floor(M/2) - sy

    E1 = np.exp(1j * -2 * np.pi * (ay/m) * np.outer(Y, V).T)
    E2 = np.exp(1j * -2 * np.pi * (ax/m) * np.outer(X, U))

    F = E1.dot(f).dot(E2)

    if unitary is True:
        norm_coef = np.sqrt((ay * ax)/(m * n * M * N))
        return F * norm_coef
    else:
        return F

def is_power_of_2(value):
    '''
    c++ inspired implementation, see SO:
    https://stackoverflow.com/questions/29480680/finding-if-a-number-is-a-power-of-2-using-recursion
    '''
    return bool(value and not value&(value-1))
