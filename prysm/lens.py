'''A base Lens class which contains the system's first-order properties
'''

class Lens(object):
    '''
    Represents an optical system.  It is defined by its first order properties
    '''
    def __init__(self, efl=1, fno=1, pupil_magnification=1):
        # for a pupil_magnification less than 1, the exit pupil is smaller than the entrance pupil
        if efl < 0:
            raise UserWarning('''\
                negative focal lengths are treated as positive for fresnel diffraction
                propogation to function correctly.  In the contself.center_pupil of these simulations
                a positive and negative focal length are functionally equivalent and the
                provide value has had its sign flipped.
                ''')
            efl *= -1
        if fno < 0:
            raise ValueError('f/# must by definition be positive')
        
        self.efl = efl
        self.fno = fno
        self.pupil_magnification = pupil_magnification
        self.epd = efl / fno
        self.xpd = self.epd * pupil_magnification

    def __repr__(self):
        return (f'Lens with properties:\n\t'
            f'efl: {self.efl}\n\t'
            f'f/#: {self.fno}\n\t'
            f'pupil mag: {self.pupil_magnification}')
