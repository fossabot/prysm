''' Configuration for this instance of prysm
'''
import numpy as np

_precision = None
_precision_complex = None

class Config(object):
    ''' global configuration of prysm.
    '''
    def __init__(self, dtype=np.float64):
        '''Tells prysm to use a given precision

        Args:
            dtype (:class:`numpy.dtype`): a valid numpy datatype.
                Should be a half, full, or double precision float.

        Returns:
            Null

        '''
        global _precision
        global _precision_complex

        if dtype not in (np.float32, np.float64):
            raise ValueError('invalid precision.  Datatype should be np.float32/64.')
        _precision = dtype
        if dtype is np.float32:
            _precision_complex = np.complex64
        else:
            _precision_complex = np.complex128
        return

    def set_precision(self, dtype):
        global _precision
        global _precision_complex
        
        if dtype not in (np.float32, np.float64):
            raise ValueError('invalid precision.  Datatype should be np.float32/64.')
        _precision = dtype
        if dtype is np.float32:
            _precision_complex = np.complex64
        else:
            _precision_complex = np.complex128
        return

    @property
    def precision(self):
        global _precision
        return _precision

    @property
    def precision_complex(self):
        global _precision_complex
        return _precision_complex

config = Config()
