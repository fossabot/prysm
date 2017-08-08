'''configuration for this instance of code6
'''
import numpy as np

class Config(object):
    ''' global configuration of code6.
    '''
    def __init__(self, dtype=np.float64):
        '''Tells code6 to use a given precision

        Args:
            dtype (:class:`numpy.dtype`): a valid numpy datatype.
                Should be a half, full, or double precision float.

        Returns:
            Null

        '''
        if dtype not in (np.float32, np.float64):
            raise ValueError('invalid precision.  Datatype should be np.float32/64.')
        self.precision = dtype
        if dtype is np.float32:
            self.precision_complex = np.complex64
        else:
            self.precision_complex = np.complex128
        return

    def set_precision(dtype):
        if dtype not in (np.float32, np.float64):
            raise ValueError('invalid precision.  Datatype should be np.float32/64.')
        self.precision = dtype
        if dtype is np.float32:
            self.precision_complex = np.complex64
        else:
            self.precision_complex = np.complex128
        return

config = Config()
