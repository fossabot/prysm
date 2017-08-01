'''Model surface finish errors of optical systems (low-amplitude, random phase errors)
'''

import numpy as np

from code6.pupil import Pupil

class SurfaceFinish(Pupil):
    '''random, normally distributed phase errors in a pupil
    '''

    def __init__(self, *args, **kwargs):
        '''
        Takes only normal Pupil arguments, and an "amplitude" keyword for scale
        '''
        self.normalize = False
        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key.lower() in ('amplitude', 'amp'):
                    self.amplitude = value
                #elif key in ('rms_norm'):
                #    self.normalize = True
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        # generate the coordinate grid so we conform to pupil spec
        self._gengrid()

        # fill the phase with random, normally distributed values, 
        # normalize to unit PV, and scale to appropriate amplitude
        self.phase = np.random.randn(self.samples, self.samples)
        self.phase /= ((self.phase.max() - self.phase.min()) / self.amplitude)

        # convert to units of nm, um, etc
        self._correct_phase_units()
        self.fcn = np.exp(1j * 2 * np.pi / self.wavelength * self.phase)
        return self.phase, self.fcn