''' Configuration for this instance of prysm
'''
import numpy as np

_precision = 64
_precision_complex = 128
_parallel_rgb = True
_backend = 'np'
_zernike_base = 1

try:
    import pyculib
    assert pyculib  # silence pyflakes
    # _backend = 'cu'  # for now, don't enable cuda (only 2-3x perf gain)
except ImportError:
    pass


class Config(object):
    ''' global configuration of prysm.
    '''
    def __init__(self,
                 precision=_precision,
                 parallel_rgb=_parallel_rgb,
                 backend=_backend,
                 zernike_base=_zernike_base):
        '''Tells prysm to use a given precision

        Args:
            precision (`int`): 32 or 64, number of bits of precision.

            parallel_rgb (`bool`): whether to parallelize RGB computations or
                not.  This improves performance for large arrays, but may slow
                things down if arrays are relatively small due to the spinup
                time of new processes.

            backend (`string`): a supported backend.  Current options are "np"
                for numpy, or "cuda" for CUDA GPU based computation based on
                numba and pyculib.

            zernike_base (`int`): base for zernikes; start at 0 or 1.

        Returns:
            new Config instance.

        '''
        global _precision
        global _precision_complex
        global _parallel_rgb
        global _backend
        global _zernike_base

        self.set_precision(precision)
        self.set_parallel_rgb(parallel_rgb)
        self.set_backend(backend)
        self.set_zernike_base(zernike_base)

    def set_precision(self, precision):
        global _precision
        global _precision_complex

        if precision not in (32, 64):
            raise ValueError('invalid precision.  Precision should be 32 or 64.')

        if precision == 32:
            _precision = np.float32
            _precision_complex = np.complex64
        else:
            _precision = np.float64
            _precision_complex = np.complex128

    def set_parallel_rgb(self, parallel):
        global _parallel_rgb
        _parallel_rgb = parallel

    def set_backend(self, backend):
        if backend.lower() not in ('np', 'numpy', 'cu', 'cuda'):
            raise ValueError('Backend must be numpy or cuda.')

        global _backend
        if backend.lower() in ('np', 'numpy'):
            _backend = 'np'
        else:
            _backend = 'cu'

    def set_zernike_base(self, base):
        if base not in (0, 1):
            raise ValueError('By convention zernike base must be 0 or 1.')

        global _zernike_base
        _zernike_base = base

    @property
    def precision(self):
        global _precision
        return _precision

    @property
    def precision_complex(self):
        global _precision_complex
        return _precision_complex

    @property
    def parallel_rgb(self):
        global _parallel_rgb
        return _parallel_rgb

    @property
    def backend(self):
        global _backend
        return _backend

    @property
    def zernike_base(self):
        global _zernike_base
        return _zernike_base


config = Config()
