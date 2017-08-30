''' A repository of seidel aberration descriptions used to model pupils of
optical systems.
'''
from functools import lru_cache

import numpy as np
from numpy import arctan2, exp, cos, sin, pi, sqrt, nan
from numpy import power as npow

from prysm.conf import config
from prysm.pupil import Pupil

class Seidel(Pupil):
    '''Seidel pupil description

    Properties:
        Inherited from :class:`Pupil`, please see that class.

    Instance Methods:
        build: computes the phase and wavefunction for the pupil.  This method
            is automatically called by the constructor, and does not regularly
            need to be changed by the user.

    Private Instance Methods:
        none

    Static Methods:
        none

    '''

    def __init__(self, *args, **kwargs):
        '''Initializes a new :class:`Seidel` :class:`Pupil`

        Args:
            samples (`int`): number of samples across pupil diameter

            wavelength (`float`): wavelength of light, in um

            epd: (`float`): diameter of the pupil, in mm

            opd_unit (`string`): unit OPD is expressed in.  One of:
                ($\lambda$, waves, $\mu m$, microns, um, nm , nanometers)

            Wxyz (`float`): W coefficient with x=H, y=rho, z=phi dependencies
                ex: W020 - defocus, W040 - spherical, W131 - coma.

        Returns:
            Seidel:  A new :class:`Seidel` pupil instance.

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
        '''Uses the wavefront coefficients stored in this class instance to
            build a wavefront model.

        Args:
            none

        Returns:
            tuple containing:
                :class:`~numpy.ndarray` containing the phase

                :class:`~numpy.ndarray` wavefunction for the pupil

        '''
        mathexpr = 'np.zeros((self.samples, self.samples))'
        for term, coef in zip(self.eqns, self.coefs):
            mathexpr += '+' + str(coef) + '*(' + term + ')'

        # pull the field point into the namespace our expression wants
        H = self.field
        self._gengrid()
        rho, phi = self.rho, self.phi

        # compute the pupil phase and wave function
        self.phase = eval(mathexpr).astype(config.precision)
        self._correct_phase_units()
        self._phase_to_wavefunction()
        return self.phase, self.fcn

    def __repr__(self):
        return str(self.__dict__)

@lru_cache()
def wexpr_to_opd_expr(Wxxx):
    '''Converts a W notation to a string with numpy code to evaluate for pupil
        phase.

    Args:
        Wxxx (`string`): A string of the form "W000," "W131", etc.

    Returns:
        `string`:  Contains typed numpy expressions to be evaluated to return phase

    '''
    # pop the W off and separate the characters
    _ = list(Wxxx[1:])
    H, rho, phi = _[0], _[1], _[2]
    # .format converts to bytecode, f-strings do not.  Micro-optimization here
    return 'H**{0} * rho**{1} * cos(phi)**{2}'.format(H, rho, phi)
