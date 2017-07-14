'''
A repository of seidel aberration descriptions used to model pupils of
optical systems.
'''
import numpy as np
from numpy import arctan2, exp, cos, sin, pi, sqrt, nan
from numpy import power as npow

from code6.pupil import Pupil

_names = [
    'W000 - Piston',
    'W111 - Tilt',
    'W020 - Defocus',
    'W040 - 3rd order Spherical',
    'W060 - 5th order Spherical',
    'W080 - 7th order Spherical',
    'W131 - 3rd order Coma',
    'W151 - 5th order Coma',
    'W171 - 7th order Coma',
    'W222 - 3rd order Astigmatism',
    'W224 - 5th order Astigmatism',
    'W226 - 7th order Astigmatism',
]
_termnos = [
    'W000',
    'W111',
    'W020',
    'W040',
    'W060',
    'W080',
    'W131',
    'W151',
    'W171',
    'W222',
    'W224',
    'W226'
]

# see fringezernike.py -- equations are stored as text and will be
# eval'd to produce the pupil model
_eqns_pupil = [
    'np.ones((self.samples, self.samples))', # W000
    'rho * cos(phi)',                        # W111
    'npow(rho,2)',                           # W020
    'npow(rho,4)',                           # W040
    'npow(rho,6)',                           # W060
    'npow(rho,8)',                           # W080
    'npow(rho,3) * cos(phi)',                # W131
    'npow(rho,5) * cos(phi)',                # W151
    'npow(rho,7) * cos(phi)',                # W171
    'npow(rho,2) * npow(cos(phi),2)',        # W222
    'npow(rho,2) * npow(cos(phi),2)',        # W224
    'npow(rho,2) * npow(cos(phi),2)',        # W226
]

# Seidel aberrations also have field dependence
_eqns_field = [
    '1',    # W000
    'H',    # W111
    '1',    # W020
    '1',    # W040
    '1',    # W060
    '1',    # W080
    'H',    # W131
    'H',    # W151
    'H',    # W171
    'H**2', # W222
    'H**4', # W224
    'H**6', # W226
]

class Seidel(Pupil):
    '''
    A pupil described by a set of Seidel coefficients
    '''

    def __init__(self, *args, **kwargs):
        '''
        kwargs: pass a named set of Seidel terms. e.g. Seidel(W040=1, W020=-0.5)
                Terms of arbitrary order can be used.
        '''

        self.eqns = []
        self.coefs = []
        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'w':
                    expr = wexpr_to_opd_expr(key)
                    self.eqns.append(expr)
                    self.coefs.append(value)
                elif key.lower() in ('field', 'relative_field', 'h'):
                    self.field = value
                else:
                    pass_args[key] = value

        if not hasattr(self, 'field'):
            self.field = 1

        super().__init__(**pass_args)

    def build(self, wavelength=0.5, relative_image_height=1):
        # construct an equation for the phase of the pupil
        mathexpr = 'np.zeros((self.samples, self.samples))'
        for term, coef in zip(self.eqns, self.coefs):
            mathexpr += '+' + str(coef) + '*(' + term + ')'

        # pull the field point into the namespace our expression wants
        H = self.field
        self._gengrid()
        rho, phi = self.rho, self.phi

        # compute the pupil phase and wave function
        self.phase = eval(mathexpr)
        self.fcn = exp(1j * 2 * pi / wavelength * self.phase)
        return self.phase, self.fcn

def wexpr_to_opd_expr(Wxxx):
    # pop the W off and separate the characters
    _ = list(Wxxx[1:])
    H, rho, phi = _[0], _[1], _[2]
    # .format converts to bytecode, f-strings do not.  Micro-optimization here
    return 'npow(H,{0}) * npow(rho,{1}) * npow(cos(phi),{2})'.format(H, rho, phi)
