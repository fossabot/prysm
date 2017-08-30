''' Model of optical systems
'''
import warnings
from functools import partial

import numpy as np
from scipy.optimize import minimize

from prysm.conf import config
from prysm.seidel import Seidel
from prysm.psf import PSF
from prysm.otf import MTF
from prysm.util import share_fig_ax

class Lens(object):
    ''' Represents a lens or optical system.
    '''
    def __init__(self, **kwargs):
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
        efl = 1
        fno = 1
        pupil_magnification = 1
        ab = dict()
        fields = [0, 1]
        fov_x = 0
        fov_y = 21.64
        fov_unit = 'mm'
        samples = 128
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
                elif kl == 'samples':
                    samples = value

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
        self.samples = samples

    ####### adjustments --------------------------------------------------------

    def autofocus(self, field_index=0):
        ''' Adjusts the W020 aberration coefficient to maximize the MTF at a
            given field index.

        Args:
            field_index (`int`): index of the field to maximize MTF at.

        Returns:
            `Lens`: self.
        '''
        coefs = self.aberrations.copy()
        try:
            # try to access the W020 aberration
            float(coefs['W020'])
        except Exception as e:
            # if it is not set, make it 0
            coefs['W020'] = 0.0

        def opt_fcn(self, coefs, w020):
            # shift the defocus term appropriately
            abers = coefs.copy()
            abers['W020'] += w020
            pupil = Seidel(**abers, epd=self.epd, samples=self.samples, h=self.fields[field_index])

            # cost value (to be minimized) is RMS wavefront
            return pupil.rms

        opt_fcn = partial(opt_fcn, self, coefs)

        new_defocus = minimize(opt_fcn, x0=0, method='CG')
        coefs['W020'] += float(new_defocus['x'])
        self.aberrations = coefs.copy()
        return self

    ####### adjustments --------------------------------------------------------

    ####### data generation ----------------------------------------------------

    def psf_vs_field(self, num_pts):
        ''' Generates a list of PSFs as a function of field.

        Args:
            num_pts (`int`): number of points to generate a PSF for.

        Returns:
            `list` containing the PSF objects.

        '''
        self._uniformly_spaced_fields(num_pts)
        psfs = []
        for idx in range(num_pts):
            psfs.append(self._make_psf(idx))
        return psfs

    def mtf_vs_field(self, num_pts, freqs=[10, 20, 30, 40, 50]):
        ''' Generates a 2D array of MTF vs field values for the given spatial
            frequencies.

        Args:
            num_pts (`int`): Number of points to compute the MTF at.

            freqs (`iterable`): set of frequencies to compute at.

        Returns:
            `tuple` containing:

                `numpy.ndarray`: (Tan) a 3D ndnarray where the columns
                    correspond to fields and the rows correspond to spatial
                    frequencies.

                `numpy.ndarray`: (Sag) a 3D ndnarray where the columns
                    correspond to fields and the rows correspond to spatial
                    frequencies.

        '''
        self._uniformly_spaced_fields(num_pts)
        mtfs_t = np.empty((num_pts, len(freqs)))
        mtfs_s = np.empty((num_pts, len(freqs)))
        for idx in range(num_pts):
            mtf = self._make_mtf(idx)
            vals_t = mtf.exact_polar(freqs, 0)
            vals_s = mtf.exact_polar(freqs, 90)
            mtfs_t[idx, :] = vals_t
            mtfs_s[idx, :] = vals_s

        return mtfs_s, mtfs_t

    ####### data generation ----------------------------------------------------

    ####### plotting -----------------------------------------------------------

    def plot_psf_vs_field(self, num_pts, fig=None, axes=None, axlim=25):
        ''' Creates a figure showing the evolution of the PSF over the field
            of view.

        Args:
            num_pts (`int`): Number of points between (0,1) to create a PSF for

        Returns:
            `tuple` containing:

                `matplotlib.pyplot.figure`: figure containing the plots.

                `list`: the axes the plots are placed in.

        '''
        psfs = self.psf_vs_field(num_pts)
        fig, axes = share_fig_ax(fig, axes, numax=num_pts, sharex=True, sharey=True)

        for idx, (psf, axis) in enumerate(zip(psfs, axes)):
            show_labels = False
            show_colorbar = False
            if idx == 0:
                show_labels = True
            elif idx == num_pts-1:
                show_colorbar = True
            psf.plot2d(fig=fig, ax=axis, axlim=axlim,
                       show_axlabels=show_labels, show_colorbar=show_colorbar)

        fig_width = 15
        fig.set_size_inches(fig_width, fig_width/num_pts)
        fig.tight_layout()
        return fig, axes

    ####### plotting -----------------------------------------------------------

    ####### helpers ------------------------------------------------------------

    def _make_pupil(self, field_index):
        ''' Generates the pupil for a given field

        Args:
            field_index (`int`): index of the desired field in the self.fields
                iterable.

        Returns:
            `Pupil`: a pupil object.
        '''
        return Seidel(**self.aberrations,
                      epd=self.epd,
                      h=self.fields[field_index],
                      samples=self.samples)

    def _make_psf(self, field_index):
        ''' Generates the psf for a given field

        Args:
            field_index (`int`): index of the desired field in the self.fields
                iterable.

        Returns:
            `PSF`: a psf object.
        '''
        p = self._make_pupil(field_index=field_index)
        return PSF.from_pupil(p, self.efl)

    def _make_mtf(self, field_index):
        ''' Generates the mtf for a given field

        Args:
            field_index (`int`): index of the desired field in the self.fields
                iterable.

        Returns:
            `MTF`: an MTF object.
        '''
        pp = self._make_psf(field_index=field_index)
        return MTF.from_psf(pp)

    def _uniformly_spaced_fields(self, num_pts):
        ''' Changes the `fields` property to n evenly spaced points from 0~1.

        Args:
            num_pts (`int`): number of points.

        Returns:
            self.

        '''
        _ = np.arange(0, num_pts, dtype=config.precision)
        flds = _ / _.max()
        self.fields = flds
        return self

    ####### helpers ------------------------------------------------------------

    def __repr__(self):
        return (f'Lens with properties:\n\t'
                f'efl: {self.efl}\n\t'
                f'f/#: {self.fno}\n\t'
                f'pupil mag: {self.pupil_magnification}')
