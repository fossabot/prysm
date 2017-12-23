''' Model surface finish errors of optical systems (low-amplitude, random phase errors)
'''

import numpy as np

from prysm.pupil import Pupil


class SurfaceFinish(Pupil):
    '''A class for adding OPD to a pupil to represent surface finish errors

    Properties:
        Inherited from Pupil, please see that class.

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
        '''Creates a new SurfaceFinish instance

        Args:
            samples (int): number of samples across pupil diameter

            wavelength (float): wavelength of light, in um

            epd: (float): diameter of the pupil, in mm

            opd_unit (string): unit OPD is expressed in.  One of:
                ($\lambda$, waves, $\mu m$, microns, um, nm , nanometers)

            amplitude (float): Peak-to-Valley amplitude of the OPD

        Returns:
            SurfaceFinish.  A new SurfaceFinish instance

        '''
        self.normalize = False
        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key.lower() in ('amplitude', 'amp'):
                    self.amplitude = value
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
        self._gengrid()

        # fill the phase with random, normally distributed values,
        # normalize to unit PV, and scale to appropriate amplitude
        self.phase = np.random.randn(self.samples, self.samples)
        self.phase /= ((self.phase.max() - self.phase.min()) / self.amplitude)

        # convert to units of nm, um, etc
        self._correct_phase_units()
        self.fcn = np.exp(1j * 2 * np.pi / self.wavelength * self.phase)
        return self.phase, self.fcn
