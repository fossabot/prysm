''' A repository of fringe zernike aberration descriptions used to model pupils of optical systems.
'''
import numpy as np
from numpy import arctan2, exp, cos, sin, pi, sqrt, nan
from numpy import power as npow

from prysm.conf import config
from prysm.pupil import Pupil

_names = [
    'Z0  - Piston / Bias',
    'Z1  - Tilt X',
    'Z2  - Tilt Y',
    'Z3  - Defocus / Power',
    'Z4  - Primary Astigmatism 00deg',
    'Z5  - Primary Astigmatism 45deg',
    'Z6  - Primary Coma X',
    'Z7  - Primary Coma Y',
    'Z8  - Primary Spherical',
    'Z9  - Primary Trefoil X',
    'Z10 - Primary Trefoil Y',
    'Z11 - Secondary Astigmatism 00deg',
    'Z12 - Secondary Astigmatism 45deg',
    'Z13 - Secondary Coma X',
    'Z14 - Secondary Coma Y',
    'Z15 - Secondary Spherical',
    'Z16 - Primary Tetrafoil X',
    'Z17 - Primary Tetrafoil Y',
    'Z18 - Secondary Trefoil X',
    'Z19 - Secondary Trefoil Y',
    'Z20 - Tertiary Astigmatism 00deg',
    'Z21 - Tertiary Astigmatism 45deg',
    'Z22 - Tertiary Coma X',
    'Z23 - Tertiary Coma Y',
    'Z24 - Tertiary Spherical',
    'Z25 - Pentafoil X',
    'Z26 - Pentafoil Y',
    'Z27 - Secondary Tetrafoil X',
    'Z28 - Secondary Tetrafoil Y',
    'Z29 - Tertiary Trefoil X',
    'Z30 - Tertiary Trefoil Y',
    'Z31 - Quarternary Astigmatism 00deg',
    'Z32 - Quarternary Astigmatism 45deg',
    'Z33 - Quarternary Coma X',
    'Z34 - Quarternary Coma Y',
    'Z35 - Quarternary Spherical',
    'Z36 - Primary Hexafoil X',
    'Z37 - Primary Hexafoil Y',
    'Z38 - Secondary Pentafoil X',
    'Z39 - Secondary Pentafoil Y',
    'Z40 - Tertiary Tetrafoil X',
    'Z41 - Tertiary Tetrafoil Y',
    'Z42 - Quaternary Trefoil X',
    'Z43 - Quaternary Trefoil Y',
    'Z44 - Quinternary Astigmatism 00deg',
    'Z45 - Quinternary Astigmatism 45deg',
    'Z46 - Quinternary Coma X',
    'Z47 - Quinternary Coma Y',
    'Z48 - Quarternary Spherical',
]

# these equations are stored as text, we will concatonate all of the strings later and use eval
# to calculate the function over the rho,phi coordinate grid.  Many regard eval as unsafe or bad
# but here there is considerable performance benefit to not iterate over a large 2D array
# multiple times, and we are guaranteed safety since we have typed the equations properly and
# using properties to protect exposure
_eqns =  [
    'np.zeros((self.samples, self.samples))',                                                                      # Z 0
    'rho * cos(phi)',                                                                                              # Z 1
    'rho * sin(phi)',                                                                                              # Z 2
    '2 * npow(rho,2) - 1',                                                                                         # Z 3
    'npow(rho,2) * cos(2*phi)',                                                                                    # Z 4
    'npow(rho,2) * sin(2*phi)',                                                                                    # Z 5
    'rho * (-2 + 3 * npow(rho,2)) * cos(phi)',                                                                     # Z 6
    'rho * (-2 + 3 * npow(rho,2)) * sin(phi)',                                                                     # Z 7
    '-6 * npow(rho,2) + 6 * npow(rho,4) + 1',                                                                      # Z 8
    'npow(rho,3) * cos(3*phi)',                                                                                    # Z 9
    'npow(rho,3) * sin(3*phi)',                                                                                    # Z10
    'npow(rho,2) * (-3 + 4 * npow(rho,2)) * cos(2*phi)',                                                           # Z11
    'npow(rho,2) * (-3 + 4 * npow(rho,2)) * sin(2*phi)',                                                           # Z12
    'rho * (3 - 12 * npow(rho,2) + 10 * npow(rho,4)) * cos(phi)',                                                  # Z13
    'rho * (3 - 12 * npow(rho,2) + 10 * npow(rho,4)) * sin(phi)',                                                  # Z14
    '12 * npow(rho,2) - 30 * npow(rho,4) + 20 * npow(rho,6) - 1',                                                  # Z15
    'npow(rho,4) * cos(4*phi)',                                                                                    # Z16
    'npow(rho,4) * sin(4*phi)',                                                                                    # Z17
    'npow(rho,3) * (-4 + 5 * npow(rho,2)) * cos(3*phi)',                                                           # Z18
    'npow(rho,3) * (-4 + 5 * npow(rho,2)) * sin(3*phi)',                                                           # Z19
    'npow(rho,2) * (6 - 20 * npow(rho,2) + 15 * npow(rho,4)) * cos(2*phi)',                                        # Z20
    'npow(rho,2) * (6 - 20 * npow(rho,2) + 15 * npow(rho,4)) * sin(2*phi)',                                        # Z21
    'rho * (-4 + 30 * npow(rho,2) - 60 * npow(rho,4) + 35 * npow(rho, 6)) * cos(phi)',                             # Z22
    'rho * (-4 + 30 * npow(rho,2) - 60 * npow(rho,4) + 35 * npow(rho, 6)) * sin(phi)',                             # Z23
    '-20 * npow(rho,2) + 90 * npow(rho,4) - 140 * npow(rho,6) + 70 * npow(rho,8) + 1',                             # Z24
    'npow(rho,5) * cos(5*phi)',                                                                                    # Z25
    'npow(rho,5) * sin(5*phi)',                                                                                    # Z26
    '(6 * npow(rho,6) - 5 * npow(rho,4)) * cos(4*phi)',                                                            # Z27
    '(6 * npow(rho,6) - 5 * npow(rho,4)) * sin(4*phi)',                                                            # Z28
    'npow(rho,3) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * cos(3*phi)',                                       # Z29
    'npow(rho,3) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * sin(3*phi)',                                       # Z30
    'npow(rho,2) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * cos(2*phi)',                                       # Z31
    'npow(rho,2) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * sin(2*phi)',                                       # Z32
    ''' rho *
        (5 - 60 * npow(rho,2) + 210 * npow(rho,4) - 280 * npow(rho,6) + 126 * npow(rho,8))
        * cos(phi)''',                                                                                             # Z33
    ''' rho *
        (5 - 60 * npow(rho,2) + 210 * npow(rho,4) - 280 * npow(rho,6) + 126 * npow(rho,8))
        * sin(phi) ''',                                                                                            # Z34
    ''' 30 * npow(rho,2)
        - 210 * npow(rho,4)
        + 560 * npow(rho,6)
        - 630 * npow(rho,8)
        + 252 * npow(rho,10)
        - 1 ''',                                                                                                   # Z35
    'npow(rho,6) * cos(6*phi)',                                                                                    # Z36
    'npow(rho,6) * sin(6*phi)',                                                                                    # Z37
    '(7 * npow(rho,7) - 6 * npow(rho,5)) * cos(5*phi)',                                                            # Z38
    '(7 * npow(rho,7) - 6 * npow(rho,5)) * sin(5*phi)',                                                            # Z39
    '(28 * npow(rho,8) - 42 * npow(rho,6) + 15 * npow(rho,4)) * cos(4*phi)',                                       # Z40
    '(28 * npow(rho,8) - 42 * npow(rho,6) + 15 * npow(rho,4)) * sin(4*phi)',                                       # Z41
    '(84 * npow(rho,9) - 168 * npow(rho,7) + 105 * npow(rho,5) - 20 * npow(rho,3)) * cos(3*phi)',                  # Z41
    '(84 * npow(rho,9) - 168 * npow(rho,7) + 105 * npow(rho,5) - 20 * npow(rho,3)) * sin(3*phi)',                  # Z43
    '''(210 * npow(rho,10) - 504 * npow(rho,8) + 420 * npow(rho,6) - 140 * npow(rho,4) + 15 * npow(rho,2))
        * cos(2*phi)''',                                                                                           # Z44
    '''(210 * npow(rho,10) - 504 * npow(rho,8) + 420 * npow(rho,6) - 140 * npow(rho,4) + 15 * npow(rho,2))
        * sin(2*phi)''',                                                                                           # Z45
    '''(462 * npow(rho, 11) - 1260 * npow(rho,9) + 1260 * npow(rho,7) - 560 * npow(rho,5) +
        105 * npow(rho,3) - 6 * rho) * cos(phi)''',                                                                # Z46
    '''(462 * npow(rho, 11) - 1260 * npow(rho,9) + 1260 * npow(rho,7) - 560 * npow(rho,5) +
        105 * npow(rho,3) - 6 * rho) * sin(phi)''',                                                                # Z47
    '''924 * npow(rho,12)
       - 2772 * npow(rho,10)
       + 3150 * npow(rho,8)
       - 1680 * npow(rho,6)
       + 420 * npow(rho,4)
       - 42 * npow(rho,2)
       + 1 ''',                                                                                                    # Z48
]

# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf
_normalizations = [
    '1',         # Z 0
    '2',         # Z 1
    '2',         # Z 2
    'sqrt(3)',   # Z 3
    'sqrt(6)',   # Z 4
    'sqrt(6)',   # Z 5
    '2*sqrt(2)', # Z 6
    '2*sqrt(2)', # Z 7
    'sqrt(5)',   # Z 8
    '2*sqrt(2)', # Z 9
    '2*sqrt(2)', # Z10
    'sqrt(10)',  # Z11
    'sqrt(10)',  # Z12
    '2*sqrt(3)', # Z13
    '2*sqrt(3)', # Z14
    'sqrt(7)',   # Z15
    'sqrt(10)',  # Z16
    'sqrt(10)',  # Z17
    '2*sqrt(3)', # Z18
    '2*sqrt(3)', # Z19
    'sqrt(14)',  # Z20
    'sqrt(14)',  # Z21
    '4',         # Z22
    '4',         # Z23
    '3',         # Z24
    '2*sqrt(3)', # Z25
    '2*sqrt(3)', # Z26
    'sqrt(14)',  # Z27
    'sqrt(14)',  # Z28
    '4',         # Z29
    '4',         # Z30
    '3*sqrt(2)', # Z31
    '3*sqrt(2)', # Z32
    '2*sqrt(5)', # Z33
    '2*sqrt(5)', # Z34
    'sqrt(11)',  # Z35
    'sqrt(14)',  # Z36
    'sqrt(14)',  # Z37
    '4',         # Z38
    '4',         # Z39
    '3*sqrt(2)', # Z40
    '3*sqrt(2)', # Z41
    '2*sqrt(5)', # Z42
    '2*sqrt(5)', # Z43
    'sqrt(22)',  # Z44
    'sqrt(22)',  # Z45
    '2*sqrt(6)', # Z46
    '2*sqrt(6)', # Z47
    'sqrt(13)',  # Z48
]

class FringeZernike(Pupil):
    '''Fringe Zernike pupil description

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

            Zx (float): xth fringe zernike coefficient, in range [0,35], 0-base.

        Returns:
            FringeZernike.  A new :class:`FringeZernike` pupil instance.

        Notes:
            Supports multiple syntaxes:
                - args: pass coefficients as a list, including terms up to the highest given Z-index.
                        e.g. passing [1,2,3] will be interpreted as Z0=1, Z1=2, Z3=3.

                - kwargs: pass a named set of zernike terms.
                          e.g. FringeZernike(Z0=1, Z1=2, Z2=3)

            Supports normalization and unit conversion, can pass kwargs:
                - rms_norm=True: coefficients have unit rms value
                
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

        self.normalize = False
        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'z':
                    idx = int(key[1:]) # strip 'Z' from index
                    self.coefs[idx] = value
                elif key in ('rms_norm'):
                    self.normalize = True
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        '''Uses the wavefront coefficients stored in this class instance to
            build a wavefront model.

        Args:
            none

        Returns:
            (numpy.ndarray, numpy.ndarray) arrays containing the phase, and the
                wavefunction for the pupil.

        '''
        mathexpr = 'np.zeros((self.samples, self.samples))'
        if self.normalize is True:
            for term, coef, norm in zip(list(range(0,36)), self.coefs, _normalizations):
                if coef is 0:
                    pass
                else:
                    mathexpr += '+' + str(coef) + '*' + norm + '*(' + _eqns[term] + ')'
        else:
            for term, coef in enumerate(self.coefs):
                if coef is 0:
                    pass
                else:
                    mathexpr += '+' + str(coef) + '*(' + _eqns[term] + ')'

        # build a coordinate system over which to evaluate this function
        rho, phi = self._gengrid()

        # compute the pupil phase and wave function
        self.phase = eval(mathexpr).astype(config.precision)
        self._correct_phase_units()
        self.fcn = exp(1j * 2 * pi / self.wavelength * self.phase, dtype=config.precision_complex)
        return self.phase, self.fcn

    def __repr__(self):
        '''Pretty-print pupil description
        '''
        if self.normalize is True:
            header = 'rms normalized Fringe Zernike description with:\n\t'
        else:
            header = 'Fringe Zernike description with:\n\t'

        strs = []
        for coef, name in zip(self.coefs, _names):
            _ = f'{coef:.3f}'
            strs.append(' '.join([_, name]))
        body = '\n\t'.join(strs)

        footer = f'\n\t{self.pv:.3f} PV, {self.rms:.3f} RMS'
        return f'{header}{body}{footer}'

def fit(data, num_terms=36, normalize=False):
    '''Fits a number of zernike coefficients to provided data by minimizing 
    the root sum square between each coefficient and the given data.  The data
    should be uniformly sampled in an x,y grid

    Args:
        num_terms (int): number of terms to fit, fits terms 0~num_terms
        normalize (bool): if true, normalize coefficients to unit RMS value

    Returns:
        numpy.ndarray.  An array of coefficients matching the input data.
    
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

    if normalize:
        norms_raw = _normalizations[0:num_terms]
        norms = np.asarray([eval(norm) for norm in norms_raw])
        coefficients = np.asarray(coefficients)
        return norms * coefficients
    else:
        return np.asarray(coefficients)
