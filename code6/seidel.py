'''
A repository of seidel aberration descriptions used to model pupils of
optical systems.
'''
import numpy as np
from numpy import arctan2, exp, cos, sin, pi, sqrt, nan
from numpy import power as npow

from code6.pupil import Pupil

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
        self.field = 1
        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'w' and len(key) == 4:
                    self.eqns.append(wexpr_to_opd_expr(key))
                    self.coefs.append(value)
                elif key.lower() in ('field', 'relative_field', 'h'):
                    self.field = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
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
        self._correct_phase_units()
        self.fcn = exp(1j * 2 * pi / self.wavelength * self.phase)
        return self.phase, self.fcn

def wexpr_to_opd_expr(Wxxx):
    # pop the W off and separate the characters
    _ = list(Wxxx[1:])
    H, rho, phi = _[0], _[1], _[2]
    # .format converts to bytecode, f-strings do not.  Micro-optimization here
    return 'npow(H,{0}) * npow(rho,{1}) * npow(cos(phi),{2})'.format(H, rho, phi)
