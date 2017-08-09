'''A repository of standard zernike aberration descriptions used to model pupils of
optical systems.
'''
import numpy as np
from numpy import arctan2, exp, cos, sin, pi, sqrt, nan
from numpy import power as npow

from code6.pupil import Pupil

_names = [
    'Z0  - Piston / Bias',
    'Z1  - Tilt X',
    'Z2  - Tilt Y',
    'Z3  - Primary Astigmatism 00deg',
    'Z4  - Defocus / Power',
    'Z5  - Primary Astigmatism 45deg',
    'Z6  - Primary Trefoil X',
    'Z7  - Primary Coma X',
    'Z8  - Primary Coma Y',
    'Z9  - Primary Trefoil Y',
    'Z10 - Primary Tetrafoil X',
    'Z11 - Secondary Astigmatism 00deg',
    'Z12 - Primary Spherical',
    'Z13 - Secondary Astigmatism 45deg',
    'Z14 - Primary Tetrafoil Y',
    'Z15 - Primary Pentafoil X',
    'Z16 - Secondary Trefoil X',
    'Z17 - Secndary Coma X',
    'Z18 - secondary Coma Y',
    'Z19 - Secondary Trefoil Y',
    'Z20 - Primary Pentafoil Y',
    'Z21 - Primary Hexafoil X',
    'Z22 - Secondary Tetrafoil X',
    'Z23 - Tertiary Astigmatism 00deg',
    'Z24 - Secondary Spherical',
    'Z25 - Tertariary Astigmatism 45deg',
    'Z26 - Secondary Tetrafoil Y',
    'Z27 - Primary Hexafoil Y',
    'Z28 - Primary Heptafoil X',
    'Z29 - Secondary Pentafoil X',
    'Z30 - Tertiary Trefoil X',
    'Z31 - Tertiary Coma X',
    'Z32 - Tertiary Coma Y',
    'Z33 - Tertiary Trefoil Y',
    'Z34 - Secondary Pentafoil Y',
    'Z35 - Primary Heptafoil Y',
    'Z36 - Primary Octafoil X',
    'Z37 - Secondary Hexafoil X',
    'Z38 - Tertiary Tetrafoil X',
    'Z39 - Quarternary Astigmatism 00deg',
    'Z40 - Tertiary Spherical',
    'Z41 - Quarternary Astigmatism 45deg',
    'Z42 - Tertiary Tetrafoil Y',
    'Z43 - Secondary Hexafoil Y',
    'Z44 - Primary Octafoil Y',
    'Z45 - Primary Nonafoil X',
    'Z46 - Secondary Heptafoil X',
    'Z47 - Tertiary Pentafoil X',
]

# these equations are stored as text, we will concatonate all of the strings later and use eval
# to calculate the function over the rho,phi coordinate grid.  Many regard eval as unsafe or bad
# but here there is considerable performance benefit to not iterate over a large 2D array
# multiple times, and we are guaranteed safety since we have typed the equations properly and
# using properties to protect exposure
_eqns =  [
    'np.ones((self.samples, self.samples))',                                                                        # Z0
    'rho * cos(phi)',                                                                                               # Z1
    'rho * sin(phi)',                                                                                               # Z2
    'npow(rho,2) * cos(2*phi)',                                                                                     # Z3
    '2 * npow(rho,2) - 1',                                                                                          # Z4
    'npow(rho,2) * sin(2*phi)',                                                                                     # Z5
    'npow(rho,3) * cos(3*phi)',                                                                                     # Z6
    '(3 * npow(rho,3) - 2 * rho) * cos(phi)',                                                                       # Z7
    '(3 * npow(rho,3) - 2 * rho) * sin(phi)',                                                                       # Z8
    'npow(rho, 3) * sin(3*phi)',                                                                                    # Z9
    'npow(rho, 4) * cos(4*phi)',                                                                                    #Z10
    '(4 * npow(rho,4) - 3 * npow(rho,2)) * cos(2*phi)',                                                             #Z11
    '-6 * npow(rho,2) + 6 * npow(rho,4) + 1',                                                                       #Z12
    '(4 * npow(rho, 4) - 3 * npow(rho, 2)) * sin(2*phi)',                                                           #Z13
    'npow(rho,4) * sin(4*phi)',                                                                                     #Z14
    'npow(rho,5) * cos(5*phi)',                                                                                     #Z15
    '(5 * npow(rho,5) - 4 * npow(rho,3)) * cos(3*phi)',                                                             #Z16
    '(10 * npow(rho,5) - 12 * npow(rho, 3) + 3 * rho) * cos(phi)',                                                  #Z17
    '(10 * npow(rho,5) - 12 * npow(rho, 3) + 3 * rho) * sin(phi)',                                                  #Z18
    '(5 * npow(rho, 5) - 4 * npow(rho, 3)) * sin(3*phi)',                                                           #Z19
    'npow(rho, 5) * cos(5*phi)',                                                                                    #Z20
    'npow(rho, 6) * cos(6*phi)',                                                                                    #Z21
    '(6 * npow(rho,6) - 5 * npow(rho,4)) * cos(4*phi)',                                                             #Z22
    '(15 * npow(rho,6) - 20 * npow(rho, 4) + 6 * npow(rho, 2)) * cos(2*phi)',                                       #Z23
    '20 * npow(rho,6) - 30 * npow(rho,4) + 12 * npow(rho,2) - 1',                                                   #Z24
    '(15 * npow(rho,6) - 20 * npow(rho,4) + 6 * npow(rho, 2)) * sin(2*phi)',                                        #Z25
    '(6 * npow(rho,6) - 5 * npow(rho,4)) * sin(4*phi)',                                                             #Z26
    'npow(rho,6) * sin(6*phi)',                                                                                     #Z27
    'npow(rho,6) * cos(7*phi)',                                                                                     #Z28
    '(7 * npow(rho,7) - 6 * npow(rho,5)) * cos(5*phi)',                                                             #Z29
    '(21 * npow(rho,7) - 30 * npow(rho, 5) + 10 * npow(rho, 3)) * cos(3*phi)',                                      #Z30
    '(35 * npow(rho,7) - 60 * npow(rho,5) + 30 * npow(rho,3) - 4 * rho) * cos(phi)',                                #Z31
    '(35 * npow(rho,7) - 60 * npow(rho,5) + 30 * npow(rho,3) - 4 * rho) * sin(phi)',                                #Z32
    '(21 * npow(rho,7) - 30 * npow(rho,5) + 10 * npow(rho,3)) * sin(3*phi)',                                        #Z33
    '(7 * npow(rho,7) - 6 * npow(rho,5)) * sin(5*phi)',                                                             #Z34
    'npow(rho,7) * sin(7*phi)',                                                                                     #Z35
    'npow(rho,8) * cos(8*phi)',                                                                                     #Z36
    '(8 * npow(rho,8) - 7 * npow(rho,6)) * cos(6*phi)',                                                             #Z37
    '(28 * npow(rho,8) - 42 * npow(rho,6) + 15 * npow(rho,4)) * cos(4*phi)',                                        #Z38
    '(56 * npow(rho,8) - 105 * npow(rho,6) + 60 * npow(rho,4) - 10 * npow(rho,2)) * cos(2*phi)',                    #Z39
    '70 * npow(rho,8) - 140 * npow(rho,7) + 90 * npow(rho,4) - 20 * npow(rho,2) + 1',                               #Z40
    '(56 * npow(rho,8) - 105 * npow(rho,6) + 60 * npow(rho,4) - 10 * npow(rho,2)) * cos(2*phi)',                    #Z41
    '(28 * npow(rho,8) - 42 * npow(rho,6) + 15 * npow(rho,4)) * sin(4*phi)',                                        #Z42
    '(8 * npow(rho,8) - 7 * npow(rho,6)) * sin(6*phi)',                                                             #Z43
    'npow(rho,8) * sin(8*phi)',                                                                                     #Z44
    'npow(rho,9) * cos(9*phi)',                                                                                     #Z45
    '(9 * npow(rho,9) - 8 * npow(rho,7)) * cos(7*phi)',                                                             #Z46
    '(36 * npow(rho,9) - 56 * npow(rho,7) + 21 * npow(rho,5)) * cos(5*phi)',                                        #Z47
]

class StandardZernike(Pupil):
    '''Standard Zernike pupil description

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
        '''Creates a FringeZernike Pupil object

        Args:
            samples (int): number of samples across pupil diameter

            wavelength (float): wavelength of light, in um

            epd: (float): diameter of the pupil, in mm

            opd_unit (string): unit OPD is expressed in.  One of:
                ($\lambda$, waves, $\mu m$, microns, um, nm , nanometers)

            Zx (float): xth standard zernike coefficient, in range [0,47], 0-base.

        Returns:
            StandardZernike.  A new :class:`StandardZernike` pupil instance.

        Notes:
            Supports multiple syntaxes:
                - args: pass coefficients as a list, including terms up to the highest given Z-index.
                        e.g. passing [1,2,3] will be interpreted as Z0=1, Z1=2, Z3=3.

                - kwargs: pass a named set of zernike terms.
                          e.g. StandardZernike(Z0=1, Z1=2, Z2=3)

            Supports unit conversion, can pass kwarg:
                - opd_unit='nm': coefficients are expressed in units of nm

            The kwargs syntax overrides the args syntax.

        '''

        if args is not None:
            if len(args) is 0:
                self.coefs = [0] * len(_eqns)
            else:
                self.coefs = [*args[0]]
        else:
            self.coefs = [0] * len(_eqns)

        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'z':
                    idx = int(key[1:]) # strip 'Z' from index
                    self.coefs[idx] = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        # construct an equation for the phase of the pupil
        mathexpr = 'np.zeros((self.samples, self.samples))'
        for term, coef in enumerate(self.coefs):
            if coef is 0:
                pass
            else:
                mathexpr += '+' + str(coef) + '*(' + _eqns[term] + ')'

        # build a coordinate system over which to evaluate this function
        rho, phi = self._gengrid()

        # compute the pupil phase and wave function
        self.phase = eval(mathexpr)
        self._correct_phase_units()
        self.fcn = exp(1j * 2 * pi / self.wavelength * self.phase)
        return self.phase, self.fcn

    def __repr__(self):
        '''Pretty-print pupil description
        '''
        header = 'Standard Zernike description with:\n\t'

        strs = []
        for coef, name in zip(self.coefs, _names):
            _ = f'{coef:.3f}'
            strs.append(' '.join([_, name]))
        body = '\n\t'.join(strs)

        footer = f'\n\t{self.pv:.3f} PV, {self.rms:.3f} RMS'
        return f'{header}{body}{footer}'

def fit(data, num_terms=47, normalize=False):
    '''
    fits a number of zernike coefficients to provided data by minimizing the root sum square
    between each coefficient and the given data.  The data should be uniformly
    sampled in an x,y grid
    '''
    if num_terms > len(_eqns):
        raise ValueError(f'number of terms must be less than {len(_eqns)}')
    sze = data.shape
    x, y = np.linspace(-1, 1, sze[0]), np.linspace(-1, 1, sze[1])
    xv, yv = np.meshgrid(x,y)
    rho = sqrt(npow(xv,2), npow(yv,2))
    phi = arctan2(yv, xv)

    # enforce circularity of the pupil
    data[rho > 1] = 0

    coefficients = []
    for i in range(num_terms):
        term_component = eval(_eqns[i])
        term_component[rho>1] = 0
        cm = sum(sum(data*term_component))*4/sze[0]/sze[1]/pi
        coefficients.append(cm)

    return coefficients
