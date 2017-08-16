''' Model of optical systems
'''
import warnings

class Lens(object):
    ''' 
    '''
    def __init__(self, *args, **kwargs):
        ''' Create a new Lens object.

        Args:
            efl (`float`): Effective Focal Length.

            fno (`float`): Focal Ratio.

            pupil_magnification (`float`): Ratio of exit pupil to entrance pupil
                diameter.

            aberrations (`dict`): A dictionary

            fields (`iterable`): A set of relative field points to analyze (symmetric)

            fov_x (`float`): half Field of View in X

            fov_y (`float`): half Field of View in Y

            fov_unit (`string`): unit for field of view.  mm, degrees, etc.

        '''

        if kwargs is not None:
            for key, value in kwargs.items():
                kl = key.lower()
                if kl == 'efl':
                    efl = value
                elif kl == 'fno':
                    fno = value
                elif kl == 'pupil_magnification':
                    pupil_magnification = value
                elif kl in ('aberrations', 'abers', 'abs'):
                    ab = value
                elif kl == 'fields':
                    fields = value
                elif kl == 'fov_x':
                    fov_x = value
                elif kl == 'fov_y':
                    fov_y = value
                elif kl == 'fov_unit':
                    fov_unit = value

        if efl < 0:
            warnings.warn('''
                Negative focal lengths are treated as positive for fresnel
                diffraction propogation to function correctly.  In the context
                of these simulations a positive and negative focal length are
                functionally equivalent and the provide value has had its sign
                flipped.
                ''')
            efl *= -1
        if fno < 0:
            raise ValueError('f/# must by definition be positive')

        self.efl = efl
        self.fno = fno
        self.pupil_magnification = pupil_magnification
        self.epd = efl / fno
        self.xpd = self.epd * pupil_magnification
        self.aberrations = ab
        self.fields = fields
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.fov_unit = fov_unit

    def __repr__(self):
        return (f'Lens with properties:\n\t'
            f'efl: {self.efl}\n\t'
            f'f/#: {self.fno}\n\t'
            f'pupil mag: {self.pupil_magnification}')
