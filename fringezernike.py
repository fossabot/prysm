'''
A repository of fringe zernike aberration descriptions used to model pupils of
optical systems.
'''
import numpy as np
from numpy import arctan2, exp, cos, sin, pi, sqrt, nan
from numpy import power as npow

from pupil import Pupil

_names = [
        'Z0  - Piston / Bias',
        'Z1  - Tilt X',
        'Z2  - Tilt Y',
        'Z3  - Defocus / Power',
        'Z4  - Primary Astigmatism X',
        'Z5  - Primary Astigmatism Y',
        'Z6  - Primary Coma X',
        'Z7  - Primary Coma Y',
        'Z8  - Primary Spherical',
        'Z9  - Primary Trefoil X',
        'Z10 - Primary Trefoil Y',
        'Z11 - Secondary Astigmatism X',
        'Z12 - Secondary Astigmatism Y',
        'Z13 - Secondary Coma X',
        'Z14 - Secondary Coma X',
        'Z15 - Secondary Spherical Y',
        'Z16 - Primary Tetrafoil X',
        'Z17 - Primary Tetrafoil Y',
        'Z18 - Secondary Trefoil X',
        'Z19 - Secondary Trefoil Y',
        'Z20 - Tertiary Astigmatism X',
        'Z21 - Tertiary Astigmatism Y',
        'Z22 - Tertiary Coma X',
        'Z23 - Tertiary Coma Y',
        'Z24 - Tertiary Spherical',
        'Z25 - Pentafoil X',
        'Z26 - Pentafoil Y',
        'Z27 - Secondary Tetrafoil X',
        'Z28 - Secondary Tetrafoil Y',
        'Z29 - Tertiary Trefoil X',
        'Z30 - Tertiary Trefoil Y',
        'Z31 - Quarternary Astigmatism X',
        'Z32 - Quarternary Astigmatism Y',
        'Z33 - Quarternary Coma X',
        'Z34 - Quarternary Coma Y',
        'Z35 - Quarternary Spherical',
    ]

# these equations are stored as text, we will concatonate all of the strings later and use eval
# to calculate the function over the rho,phi coordinate grid.  Many regard eval as unsafe or bad
# but here there is considerable performance benefit to not iterate over a large 2D array
# multiple times, and we are guaranteed safety since we have typed the equations properly and
# using properties to protect exposure
_eqns =  [
    'np.ones((self.samples, self.samples))',                                            # Z0
    'rho * cos(phi)',                                                                   # Z1
    'rho * sin(phi)',                                                                   # Z2
    '2 * npow(rho,2) - 1',                                                              # Z3
    'npow(rho,2) * cos(2*phi)',                                                         # Z4
    'npow(rho,2) * sin(2*phi)',                                                         # Z5
    'rho * (-2 + 3 * npow(rho,2)) * cos(phi)',                                          # Z6
    'rho * (-2 + 3 * npow(rho,2)) * sin(phi)',                                          # Z7
    '-6 * npow(rho,2) + 6 * npow(rho,4) + 1',                                           # Z8
    'npow(rho,3) * cos(3*phi)',                                                         # Z9
    'npow(rho,3) * sin(3*phi)',                                                         #Z10
    'npow(rho,2) * (-3 + 4 * npow(rho,2)) * cos(2*phi)',                                #Z11
    'npow(rho,2) * (-3 + 4 * npow(rho,2)) * sin(2*phi)',                                #Z12
    'rho * (3 - 12 * npow(rho,2) + 10 * npow(rho,4)) * cos(phi)',                       #Z13
    'rho * (3 - 12 * npow(rho,2) + 10 * npow(rho,4)) * sin(phi)',                       #Z14
    '12 * npow(rho,2) - 30 * npow(rho,4) + 20 * npow(rho,6) - 1',                       #Z15
    'npow(rho,4) * cos(4*phi)',                                                         #Z16
    'npow(rho,4) * sin(4*phi)',                                                         #Z17
    'npow(rho,3) * (-4 + 5 * npow(rho,2)) * cos(3*phi)',                                #Z18
    'npow(rho,3) * (-4 + 5 * npow(rho,2)) * sin(3*phi)',                                #Z19
    'npow(rho,2) * (6 - 20 * npow(rho,2) + 15 * npow(rho,4)) * cos(2*phi)',             #Z20
    'npow(rho,2) * (6 - 20 * npow(rho,2) + 15 * npow(rho,4)) * sin(2*phi)',             #Z21
    'rho * (-4 + 30 * npow(rho,2) - 60 * npow(rho,4) + 35 * npow(rho, 6)) * cos(phi)',  #Z22
    'rho * (-4 + 30 * npow(rho,2) - 60 * npow(rho,4) + 35 * npow(rho, 6)) * sin(phi)',  #Z23
    '-20 * npow(rho,2) + 90 * npow(rho,4) - 140 * npow(rho,6) + 70 * npow(rho,8) + 1',  #Z24
    'npow(rho,5) * cos(5*phi)',                                                         #Z25
    'npow(rho,5) * sin(5*phi)',                                                         #Z26
    'npow(rho,4) * (-5 + 6 * npow(rho,2) * cos(4*phi))',                                #Z27
    'npow(rho,4) * (-5 + 6 * npow(rho,2) * sin(4*phi))',                                #Z28
    'npow(rho,3) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * cos(3*phi)',            #Z29
    'npow(rho,3) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * sin(3*phi)',            #Z30
    'npow(rho,2) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * cos(2*phi)',            #Z31
    'npow(rho,2) * (10 - 30 * npow(rho,2) + 21 * npow(rho,4)) * sin(2*phi)',            #Z32
    ''' rho *
        (5 - 60 * npow(rho,2) + 210 * npow(rho,4) - 280 * npow(rho,6) + 126 * npow(rho,8))
        * cos(phi)''',                                                                  #Z33
    ''' rho *
        (5 - 60 * npow(rho,2) + 210 * npow(rho,4) - 280 * npow(rho,6) + 126 * npow(rho,8))
        * sin(phi) ''',                                                                 #Z34
    ''' 30 * npow(rho,2)
        - 210 * npow(rho,4)
        + 560 * npow(rho,6)
        - 630 * npow(rho,8)
        + 252 * npow(rho,10)
        - 1 ''',                                                                        #Z35
    ]

_normalizations = [
    '1/pi',                # Z1
    '2/pi',                # Z2
    '2/pi',                # Z3
    'sqrt(3/pi)',          # Z4
    'sqrt(6/pi)',          # Z5
    'sqrt(6/pi)',          # Z6
    'sqrt(8/pi)',          # Z7
    'sqrt(8/pi)',          # Z8
    'sqrt(5/pi)',          # Z9
    'sqrt(8/pi)',          # Z10
    'sqrt(8/pi)',          # Z11
    'sqrt(10/pi)',         # Z12
    'sqrt(10/pi)',         # Z13
    'sqrt(12/pi)',         # Z14
    'sqrt(7/pi)',          # Z15
    'sqrt(10/pi)',         # Z16
    'sqrt(10/pi)',         # Z17
    'sqrt(12/pi)',         # Z18
    'sqrt(12/pi)',         # Z19
    'sqrt(14/pi)',         # Z20
    'sqrt(14/pi)',         # Z21
    'sqrt(16/pi)',         # Z22
    'sqrt(16/pi)',         # Z23
    'sqrt(9/pi)',          # Z24
    'sqrt(12/pi)',         # Z25
    'sqrt(12/pi)',         # Z26
    'sqrt(14/pi)',         # Z27
    'sqrt(14/pi)',         # Z28
    'sqrt(16/pi)',         # Z29
    'sqrt(16/pi)',         # Z30
    'sqrt(18/pi)',         # Z31 -- may contain error
    'sqrt(18/pi)',         # Z32 -- may contain error
    'sqrt(20/pi)',         # Z33 -- may contain error
    'sqrt(20/pi)',         # Z34 -- may contain error
    'sqrt(11/pi)',         # Z35 -- may contain error
]

class FringeZernike(Pupil):
    '''
    A pupil described by a set of fringe zernike polynomials
    '''
    def __init__(self, *args, **kwargs):
        '''
        Supports multiple syntaxes:
            - args: pass coefficients as a list, including terms up to the highest given Z-index.
                    e.g. passing [1,2,3] will be interpreted as Z0=1, Z1=2, Z3=3.
            - kwargs: pass a named set of zernike terms.
                      e.g. FringeZernike(Z0=1, Z1=2, Z2=3)

        Supports normalization and unit conversion, can pass kwargs:
            - rms_norm=True: coefficients have unit rms value
            - 

        The kwargs syntax overrides the args syntax.
        '''
        
        if args is not None:
            if len(args) is 0:
                self.coefs = [0] * 36
            else:
                self.coefs = [*args[0]]
        else:
            self.coefs = [0] * 36

        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() is 'z':
                    idx = int(key[1:]) # strip 'Z' from index
                    self.coefs[idx] = value

        if 'rms_norm' in kwargs:
            self.normalize = bool(kwargs['rms_norm'])
        else:
            self.normalize = False
        
        super().__init__(**kwargs)

    def build(self, wavelength=0.5):
        # construct an equation for the phase of the pupil
        mathexpr = 'np.zeros((self.samples, self.samples))'
        if self.normalize is True:
            for term, coef, norm in enumerate(zip(self.coefs,_normalizations)):
                if coef is 0:
                    pass
                else:
                    mathexpr += '+' + str(coef) + '*(' + _eqns[term] + ')'
        else:
            for term, coef in enumerate(self.coefs):
                if coef is 0:
                    pass
                else:
                    mathexpr += '+' + str(coef) + '*(' + _eqns[term] + ')'

        # build a coordinate system over which to evaluate this function
        x = y    = np.linspace(-1, 1, self.samples)
        xv, yv   = np.meshgrid(x,y)
        self.rho = sqrt(npow(xv,2) + npow(yv,2))
        self.phi = arctan2(yv, xv)

        # duplicate for the eval() below
        rho, phi = self.rho, self.phi
        
        # compute the pupil phase and wave function
        self.phase = eval(mathexpr)      
        self.fcn = exp(1j * 2 * pi / wavelength * self.phase)
        return self.phase, self.fcn

def fit(data, num_terms=36, normalize=False):
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

    coefficients = []
    for i in range(num_terms):
        term_component = eval(_eqns[i])
        term_component[rho>1] = 0
        cm = sum(sum(data*term_component))*4/sze[0]/sze[1]/pi
        coefficients.append(cm)

    return coefficients
