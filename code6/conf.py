'''configuration for this instance of code6
'''
import numpy as np

# default to double precision
_PRECISION = np.float64
_PRECISION_COMPLEX = np.complex128

def set_precision(dtype):
    '''Tells code6 to use a given precision

    Args:
        dtype (:class:`numpy.dtype`): a valid numpy datatype.  
            Should be a half, full, or double precision float.

    Returns:
        Null

    '''
    if dtype not in (np.float16, np.float32, np.float64):
        raise ValueError('invalid precision.  Datatype should be np.float16/32/64.')
    _PRECISION = dtype
    if dtype is np.float16 or dtype is np.float32:
        _PRECISION_COMPLEX = np.complex64
    else:
        _PRECISION_COMPLEX = np.complex128
    return Null

def precision(type='f'):
    '''returns the precision of this instance of code6

    Args:
        type (`string): 'c' for complex, otherwise returns standard float type

    Returns:
        :class:`numpy.dtype`: data type to utilize.
    '''
    if type.lower() == 'c':
        return _PRECISION_COMPLEX
    else:
        return _PRECISION
