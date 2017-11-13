''' This submodule imports and exports math functions from different libraries.
    The intend is to make the backend for prysm interoperable, allowing users to
    utilize more high performance engines if they have them installed, or fall
    back to more widely available options in the case that they do not.
'''
from math import (
    floor,
    ceil,
    sin,
    cos,
    tan,
    pi,
    nan,
)
from numpy import (
    sqrt,
    arctan2,
    exp,
)

atan2 = arctan2

import numpy as np

from prysm.conf import config

# numba funcs, cuda
try:
    from numba import cuda, jit
except ImportError:
    # if Numba is not installed, create the jit decorator and have it return the
    # original function.
    def jit(signature_or_function=None, locals={}, target='cpu', cache=False, **options):
        return signature_or_function

from pyculib.fft import FFTPlan

###### CUDA code ---------------------------------------------------------------

def cast_array(array):
    ''' Casts an array to the appropriate complex format.

    Args:
        array: (`numpy.ndarray`): array.

    Returns:
        `numpy.ndarray`: array cast to appopriate complex type.

    '''
    if array.dtype == np.float32:
        return array.astype(np.complex64)
    else:
        return array.astype(np.complex128)

# trigger cuFFT initialization when submodule loads (takes ~1s)
FFTPlan((384, 384),
        itype=config.precision_complex,
        otype=config.precision_complex)

# create a map of output array types from input array types
arr1 = np.empty(1, dtype=np.float32)
arr2 = np.empty(1, dtype=np.float64)
arr3 = np.empty(1, dtype=np.complex64)
arr4 = np.empty(1, dtype=np.complex128)
cuda_out_map = {
    arr1.dtype: np.complex64,
    arr2.dtype: np.complex128,
    arr3.dtype: np.complex64,
    arr4.dtype: np.complex128,
}

# prepare a variable to cache FFT plans
_plans = dict()

def best_grid_size(size, tpb):
    ''' Computes the best grid size for a gpu array, given an array size and
        number of threads per block.

    Args:
        size (`tuple`): shape of the source array.

        tpb (`int`): number of threads per block.

    Returns:
        `tuple`. TODO: doc more

    '''
    bpg = np.ceil(np.array(size, dtype=np.float) / tpb).astype(np.int).tolist()
    return tuple(bpg)

def cu_fft2(array):
    ''' Executes a 2D fast fourier transform on CUDA GPUs.

    Args:
        array (`numpy.ndarray`): array of 32 or 64-bit floats, or 64 or 128 bit
            complex values.

    Returns:
        (`numpy.ndarray`): a new ndarray that is the FT of the input array
    '''

    hashstr = str(array.shape) + str(array.dtype) + str(cuda_out_map[array.dtype])

    # Cast the array to a complex one, because real -> complex ffts from cuFFT
    # follow the fftw convention of being formatted in a way that is hard to
    # understand.
    # Casting makes up 2/3 of the execution time for large arrays.  This should
    # be re-implemented in cuda.
    array = cast_array(array)
    rslt = cuda.pinned_array(array.shape, dtype=cuda_out_map[array.dtype])
    d_arr = cuda.to_device(array)
    d_rslt = cuda.to_device(rslt)

    # try to cache FFT plans for more speed.
    # hashstr and try/except block cost ~2us on i7-7700HQ CPU.
    try:
        _plans[hashstr].forward(d_arr, out=d_rslt)
    except KeyError:
        _plans[hashstr] = FFTPlan(shape=array.shape,
                                  itype=array.dtype,
                                  otype=cuda_out_map[array.dtype])
        _plans[hashstr].forward(d_arr, out=d_rslt)

    # rslt and d_dslt are pinned to each other in the cuda api.  The data will
    # transfer between them without CPU cycles being used, then python will
    # return the pointer in a few ns.  This shaves about 33% off of execution
    # time for fairly large arrays.
    d_rslt.copy_to_host(rslt)
    return rslt

def cu_ifft2(array):
    ''' Executes a 2D inverse fast fourier transform on CUDA GPUs.

    Args:
        array (`numpy.ndarray`): array of 32 or 64-bit floats, or 64 or 128 bit
            complex values.

    Returns:
        (`numpy.ndarray`): a new ndarray that is the FT of the input array
    '''

    hashstr = str(array.shape) + str(array.dtype) + str(cuda_out_map[array.dtype])
    array = cast_array(array)
    rslt = cuda.pinned_array(array.shape, dtype=cuda_out_map[array.dtype])
    d_arr = cuda.to_device(array)
    d_rslt = cuda.to_device(rslt)
    try:
        _plans[hashstr].inverse(d_arr, out=d_rslt)
    except KeyError:
        _plans[hashstr] = FFTPlan(shape=array.shape,
                                  itype=array.dtype,
                                  otype=cuda_out_map[array.dtype])
        _plans[hashstr].inverse(d_arr, out=d_rslt)
    d_rslt.copy_to_host(rslt)
    return rslt

###### CUDA code ---------------------------------------------------------------

###### export control ----------------------------------------------------------

# thanks, ITAR

def fft2(array):
    if config.backend == 'np':
        return np.fft.fft2(array)
    elif config.backend == 'cu':
        return cu_fft2(array)
    else:
        raise KeyError('Invalid backend for fft')

def ifft2(array):
    if config.backend == 'np':
        return np.fft.ifft2(array)
    elif config.backend == 'cu':
        return cu_ifft2(array)
    else:
        raise KeyError('Invalid backend for fft')
