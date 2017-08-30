''' A base point spread function interface
'''
import numpy as np
from numpy import floor
from numpy import power as npow
from numpy.fft import fftshift, ifftshift

from scipy import interpolate

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import make_rgb_axes

from prysm.conf import config
from prysm.mathops import fft2, ifft2
from prysm.fttools import pad2d, forward_ft_unit
from prysm.coordinates import uniform_cart_to_polar, resample_2d_complex
from prysm.util import pupil_sample_to_psf_sample, correct_gamma, share_fig_ax

class PSF(object):
    ''' Point Spread Function representations.

    Properties:
        slice_x: 1D slice through center of PSF along X axis.  Returns (x,y) data.

        slice_y: 1D slice through cente rof PSF along y axis.  Returns (x,y) data.

    Instance Methods:
        encircled_energy: Computes the encircled energy along the specified
            azimuth. If azimuth is none, returns the azimuthal average.
            Returned value is the average of both the negative and positive
            sides of the PSF along the specified azimuth.

        plot2d: Makes a 2D plot of the PSF.  Returns (fig, axis).

        plot_slice_xy: Makes a 1D plot with x and y slices through the center
            of the PSF.  Returns (fig, axis).

        plot_encircled_energy: Makes a 1D plot of the encircled energy at the
            specified azimuth.  Returns (fig, axis).

        conv: convolves this PSF with another.  Returns a new PSF object that is
            sampled at the same points as this PSF.

    Private Instance Methods:
        _renorm: renormalizes the PSF to unit peak intensity.

    Static Methods:
        from_pupil: given a pupil and a focal length, returns a PSF.

    Notes:
        Subclasses must implement an analyic_ft method with signature
            a_ft(unit_x, unit_y).

    '''
    def __init__(self, data, sample_spacing):
        ''' Creates a PSF object.

        Args:
            data (`numpy.ndarray`): intensity data for the PSF.

            sample_spacing (`float`): center-to-center spacing of samples,
                expressed in microns.

        Returns:
            `PSF`: a new PSF instance.

        '''

        self.data = data
        self.sample_spacing = sample_spacing
        self.samples_x, self.samples_y = data.shape
        self.center_x = self.samples_x // 2
        self.center_y = self.samples_y // 2

        # compute ordinate axis
        ext_x = self.sample_spacing * self.samples_x / 2
        ext_y = self.sample_spacing * self.samples_y / 2
        self.unit_x = np.linspace(-ext_x, ext_x-sample_spacing,
                                  self.samples_x, dtype=config.precision)
        self.unit_y = np.linspace(-ext_y, ext_y-sample_spacing,
                                  self.samples_y, dtype=config.precision)

    # quick-access slices ------------------------------------------------------

    @property
    def slice_x(self):
        ''' Retrieves a slice through the x axis of the PSF.
        '''
        return self.unit_x, self.data[self.center_x, :]

    @property
    def slice_y(self):
        ''' Retrieves a slices through the y axis of the PSF.
        '''
        return self.unit_y, self.data[:, self.center_y]


    def encircled_energy(self, azimuth=None):
        ''' Returns the encircled energy at the requested azumith.  If azimuth
            is None, returns the azimuthal average.

        Args:
            azimuth (`float`): azimuth to retrieve data along, in degrees.

        Returns:
            np.ndarray, np.ndarray.  Unit, encircled energy.

        '''

        # interp_dat is shaped with axis0=phi, axis1=rho
        rho, phi, interp_dat = uniform_cart_to_polar(self.unit_y, self.unit_x, self.data)

        if azimuth is None:
            # take average of all azimuths as input data
            dat = np.average(interp_dat, axis=0)
        else:
            index = np.searchsorted(phi, np.radians(azimuth))
            dat = interp_dat[index, :]

        enc_eng = np.cumsum(dat, dtype=config.precision)
        return self.unit_x[self.center_x:], enc_eng / enc_eng[-1]

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, log=False, axlim=25, interp_method='lanczos',
               pix_grid=None, fig=None, ax=None):
        ''' Creates a 2D plot of the PSF.

        Args:
            log (`bool`): if true, plot in log scale.  If false, plot in linear
                scale.

            axlim (`float`): limits of axis, symmetric.
                xlim=(-axlim,axlim), ylim=(-axlim, axlim).

            interp_method (string): method used to interpolate the image between
                samples of the PSF.

            pix_grid (float): if not None, overlays gridlines with spacing equal
                to pix_grid.  Intended to show the collection into camera pixels
                while still in the oversampled domain.

            fig (pyplot.figure): figure to plot in.

            ax (pyplot.axis): axis to plot in.

        Returns:
            pyplot.fig, pyplot.axis.  Figure and axis containing the plot.

        '''
        if log:
            fcn = 20 * np.log10(1e-100 + self.data)
            label_str = 'Normalized Intensity [dB]'
            lims = (-100, 0) # show first 100dB -- range from (1e-6, 1) in linear scale
        else:
            fcn = correct_gamma(self.data)
            label_str = 'Normalized Intensity [a.u.]'
            lims = (0, 1)

        left, right = self.unit_x[0], self.unit_x[-1]
        bottom, top = self.unit_y[0], self.unit_y[-1]

        fig, ax = share_fig_ax(fig, ax)

        # TODO: check bottom/top
        im = ax.imshow(fcn,
                       extent=[left, right, bottom, top],
                       origin='lower',
                       cmap='Greys_r',
                       interpolation=interp_method,
                       clim=lims)
        fig.colorbar(im, label=label_str, ax=ax, fraction=0.046)
        ax.set(xlabel=r'Image Plane X [$\mu m$]',
               ylabel=r'Image Plane Y [$\mu m$]',
               xlim=(-axlim, axlim),
               ylim=(-axlim, axlim))

        if pix_grid is not None:
            # if pixel grid is desired, add it
            mult = np.floor(axlim / pix_grid)
            gmin, gmax = -mult * pix_grid, mult*pix_grid
            pts = np.arange(gmin, gmax, pix_grid)
            ax.set_yticks(pts, minor=True)
            ax.set_xticks(pts, minor=True)
            ax.yaxis.grid(True, which='minor', color='white', alpha=0.25)
            ax.xaxis.grid(True, which='minor', color='white', alpha=0.25)

        return fig, ax

    def plot_slice_xy(self, log=False, axlim=20, fig=None, ax=None):
        ''' Makes a 1D plot of X and Y slices through the PSF.

        Args:
            log (`bool`): if true, plot in log scale.  if false, plot in linear
                scale.

            axlim (`float`): limits of axis, will plot [-axlim, axlim].

            fig (pyplot.figure): figure to plot in.

            ax (pyplot.axis): axis to plot in.

        Returns:
            pyplot.fig, pyplot.axis.  Figure and axis containing the plot.

        '''
        ux, x = self.slice_x
        uy, y = self.slice_y
        if log:
            fcn_x = 20 * np.log10(1e-100 + x)
            fcn_y = 20 * np.log10(1e-100 + y)
            label_str = 'Normalized Intensity [dB]'
            lims = (-120, 0)
        else:
            fcn_x = x
            fcn_y = y
            label_str = 'Normalized Intensity [a.u.]'
            lims = (0, 1)

        fig, ax = share_fig_ax(fig, ax)

        ax.plot(ux, fcn_x, label='Slice X', lw=3)
        ax.plot(uy, fcn_y, label='Slice Y', lw=3)
        ax.set(xlabel=r'Image Plane X [$\mu m$]',
               ylabel=label_str,
               xlim=(-axlim, axlim),
               ylim=lims)
        plt.legend(loc='upper right')
        return fig, ax

    def plot_encircled_energy(self, azimuth=None, axlim=20, fig=None, ax=None):
        ''' Makes a 1D plot of the encircled energy at the given azimuth.

        Args:
            azimuth (`float`): azimuth to plot at, in degrees.

            axlim (`float`): limits of axis, will plot [0, axlim].

            fig (pyplot.figure): figure to plot in.

            ax (pyplot.axis): axis to plot in.

        Returns:
            pyplot.fig, pyplot.axis.  Figure and axis containing the plot

        '''
        unit, data = self.encircled_energy(azimuth)

        fig, ax = share_fig_ax(fig, ax)
        ax.plot(unit, data, lw=3)
        ax.set(xlabel=r'Image Plane Distance [$\mu m$]',
               ylabel=r'Encircled Energy [Rel 1.0]',
               xlim=(0, axlim))
        return fig, ax

    # plotting -----------------------------------------------------------------

    # helpers ------------------------------------------------------------------

    def conv(self, psf2):
        '''Convolves this PSF with another

        Args:
            psf2 (`PSF`): PSf to convolve with this one.

        Returns:
            PSF:  A new `PSF` that is the convolution of these two PSFs.

        Notes:
            output PSF has equal sampling to whichever PSF has a lower nyquist
                frequency.

        '''
        try:
            psf_ft = fftshift(fft2(self.data))
            psf_unit_x = forward_ft_unit(self.sample_spacing, self.samples_x)
            psf_unit_y = forward_ft_unit(self.sample_spacing, self.samples_y)
            psf2_ft = psf2.analytic_ft(psf_unit_x, psf_unit_y)
            psf3 = PSF(data=abs(ifft2(psf_ft * psf2_ft)),
                       sample_spacing=self.sample_spacing)
            return psf3._renorm()
        except AttributeError: # no analytic FT on the PSF/subclass
            print('attrerr')
            return convpsf(self, psf2)

    def _renorm(self, to='peak'):
        ''' Renormalizes the PSF to unit peak intensity.

        Args:
            to (`string`): renormalization target.  Either "peak" or "total",
                produces a PSF of unit peak or total intensity.

        Returns:
            `PSF`: a renormalized PSF instance.

        '''
        if to.lower() == 'peak':
            self.data /= self.data.max()
        elif to.lower() == 'total':
            ttl = self.data.sum()
            self.data /= ttl
        return self

    # helpers ------------------------------------------------------------------

    @staticmethod
    def from_pupil(pupil, efl, padding=1):
        ''' Uses scalar diffraction propogation to generate a PSF from a pupil.

        Args:
            pupil (prysm.Pupil): Pupil, with OPD data and wavefunction.

            efl (float): effective focal length of the optical system.

            padding (number): number of pupil widths to pad each side of the
                pupil with during computation.

        Returns:
            PSF.  A new PSF instance.

        '''
        # padded pupil contains 1 pupil width on each side for a width of 3
        psf_samples = (pupil.samples * padding) * 2 + pupil.samples
        sample_spacing = pupil_sample_to_psf_sample(pupil_sample=pupil.sample_spacing * 1000,
                                                    num_samples=psf_samples,
                                                    wavelength=pupil.wavelength,
                                                    efl=efl)
        padded_wavefront = pad2d(pupil.fcn, padding)
        impulse_response = ifftshift(fft2(fftshift(padded_wavefront)))
        psf = abs(impulse_response)**2
        return PSF(psf / np.max(psf), sample_spacing)

class MultispectralPSF(PSF):
    ''' A PSF which includes multiple wavelength components.
    '''
    def __init__(self, psfs, weights):
        ''' Creates a new :class:`MultispectralPSF` instance.

        Args:
            psfs (iterable): iterable of PSFs.

            weights (iterable): iterable of weights associated with each PSF.

        Returns:
            MultispectralPSF.  A new `MultispectralPSF`.

        '''

        # find the most densely sampled PSF
        min_spacing = 1e99
        ref_idx = None
        ref_unit_x = None
        ref_unit_y = None
        ref_samples_x = None
        ref_samples_y = None
        for idx, psf in enumerate(psfs):
            if psf.sample_spacing < min_spacing:
                min_spacing = psf.sample_spacing
                ref_idx = idx
                ref_unit_x = psf.unit_x
                ref_unit_y = psf.unit_y
                ref_samples_x = psf.samples_x
                ref_samples_y = psf.samples_y

        merge_data = np.zeros((ref_samples_y, ref_samples_x, len(psfs)))
        for idx, psf in enumerate(psfs):
            # don't do anything to our reference PSF
            if idx is ref_idx:
                merge_data[:, :, idx] = psf.data * weights[idx]
            else:
                xv, yv = np.meshgrid(ref_unit_x, ref_unit_y)
                interpf = interpolate.RegularGridInterpolator((psf.unit_y, psf.unit_x), psf.data)
                merge_data[:, :, idx] = interpf((yv, xv), method='linear') * weights[idx]

        self.weights = weights
        super().__init__(merge_data.sum(axis=2), min_spacing)
        self._renorm()

class RGBPSF(object):
    ''' Trichromatic PSF, intended to show chromatic aberrations.
    '''
    def __init__(self, r_psf, g_psf, b_psf):
        ''' Creates a new `RGBPSF` instance.

        Args:
            r_psf (`PSF`): PSF for the red channel.

            g_psf (`PSF`): PSF for the green channel.

            b_psf (`PSF`): PSF for the blue channel.

        Returns:
            RGBPSF: A new `RGBPSF` instance.

        '''
        if ( np.array_equal(r_psf.unit_x, g_psf.unit_x) and \
             np.array_equal(g_psf.unit_x, b_psf.unit_x) and \
             np.array_equal(r_psf.unit_y, g_psf.unit_y) and \
             np.array_equal(g_psf.unit_y, b_psf.unit_y) ):
            # do not need to interpolate the arrays
            self.R = r_psf.data
            self.G = g_psf.data
            self.B = b_psf.data
        else:
            # need to interpolate the arrays.  Blue tends to be most densely
            # sampled, use it to define our grid
            self.B = b_psf.data

            xv, yv = np.meshgrid(b_psf.unit_x, b_psf.unit_y)
            interpf_r = interpolate.RegularGridInterpolator((r_psf.unit_y, r_psf.unit_x), r_psf.data)
            interpf_g = interpolate.RegularGridInterpolator((g_psf.unit_y, g_psf.unit_x), g_psf.data)
            self.R = interpf_r((yv, xv), method='linear')
            self.G = interpf_g((yv, xv), method='linear')

        self.sample_spacing = b_psf.sample_spacing
        self.samples_x = b_psf.samples_x
        self.samples_y = b_psf.samples_y
        self.unit_x = b_psf.unit_x
        self.unit_y = b_psf.unit_y
        self.center_x = b_psf.center_x
        self.center_y = b_psf.center_y

    def _renorm(self, to='peak'):
        ''' Renormalizes the PSF to unit peak intensity.

        Args:
            to (`string`): renormalization target.  Either "peak" or "total",
                produces a PSF of unit peak or total intensity.

        Returns:
            `PSF`: a renormalized PSF instance.

        '''
        if to.lower() == 'peak':
            self.R /= self.R.max()
            self.G /= self.G.max()
            self.B /= self.B.max()
        elif to.lower() == 'total':
            # scale to energy of the green channel.  First, make all unit peak
            self.R /= self.R.max()
            self.G /= self.G.max()
            self.B /= self.B.max()

            # compute energy of green, scale all to this value
            ttl = self.G.sum()
            self.R /= ttl
            self.G /= ttl
            self.B /= ttl
        return self

    @property
    def r_psf(self):
        return PSF(self.R, self.sample_spacing)

    @property
    def g_psf(self):
        return PSF(self.G, self.sample_spacing)

    @property
    def b_psf(self):
        return PSF(self.B, self.sample_spacing)

    def plot2d(self, log=False, axlim=25, interp_method='lanczos',
               pix_grid=None, fig=None, ax=None):
        ''' Creates a 2D color plot of the PSF.

        Args:
            log (`bool`): if true, plot in log scale.  If false, plot in linear
                scale.

            axlim (`float`): limits of axis, symmetric.
                xlim=(-axlim,axlim), ylim=(-axlim, axlim).

            interp_method (`string`): method used to interpolate the image between
                samples of the PSF.

            pix_grid (`float`): if not None, overlays gridlines with spacing equal
                to pix_grid.  Intended to show the collection into camera pixels
                while still in the oversampled domain.

            fig (pyplot.figure): figure to plot in.

            ax (pyplot.axis): axis to plot in.

        Returns:
            pyplot.fig, pyplot.axis.  Figure and axis containing the plot.

        Notes:
            Largely a copy-paste of plot2d() from the PSF class.  Some  refactoring
                could be done to make the code more succinct and unified.

        '''
        dat = np.empty((self.samples_x, self.samples_y, 3))
        dat[:, :, 0] = self.R
        dat[:, :, 1] = self.G
        dat[:, :, 2] = self.B

        if log:
            fcn = 20 * np.log10(1e-100 + dat)
            label_str = 'Normalized Intensity [dB]'
            lims = (-100, 0) # show first 100dB -- range from (1e-6, 1) in linear scale
        else:
            fcn = correct_gamma(dat)
            label_str = 'Normalized Intensity [a.u.]'
            lims = (0, 1)

        left, right = self.unit_x[0], self.unit_x[-1]
        bottom, top = self.unit_y[0], self.unit_y[-1]

        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(fcn,
                       extent=[left, right, bottom, top],
                       interpolation=interp_method,
                       origin='lower')
        ax.set(xlabel=r'Image Plane X [$\mu m$]',
               ylabel=r'Image Plane Y [$\mu m$]',
               xlim=(-axlim, axlim),
               ylim=(-axlim, axlim))

        if pix_grid is not None:
            # if pixel grid is desired, add it
            mult = np.floor(axlim / pix_grid)
            gmin, gmax = -mult * pix_grid, mult*pix_grid
            pts = np.arange(gmin, gmax, pix_grid)
            ax.set_yticks(pts, minor=True)
            ax.set_xticks(pts, minor=True)
            ax.yaxis.grid(True, which='minor')
            ax.xaxis.grid(True, which='minor')

        return fig, ax

    def plot2d_rgbgrid(self, axlim=25, interp_method='lanczos',
                       pix_grid=None, fig=None, ax=None):
        '''Creates a 2D color plot of the PSF and components

        Args:
            axlim (`float`): limits of axis, symmetric.
                xlim=(-axlim,axlim), ylim=(-axlim, axlim).

            interp_method (string): method used to interpolate the image between
                samples of the PSF.

            pix_grid (float): if not None, overlays gridlines with spacing equal
                to pix_grid.  Intended to show the collection into camera pixels
                while still in the oversampled domain.

            fig (pyplot.figure): figure to plot in.

            ax (pyplot.axis): axis to plot in.

        Returns:
            pyplot.fig, pyplot.axis.  Figure and axis containing the plot.

        Notes:
            Need to refine internal workings at some point.

        '''

        # make the arrays for the RGB images
        dat = np.empty((self.samples, self.samples, 3))
        datr = np.zeros((self.samples, self.samples, 3))
        datg = np.zeros((self.samples, self.samples, 3))
        datb = np.zeros((self.samples, self.samples, 3))
        dat[:, :, 0] = self.R
        dat[:, :, 1] = self.G
        dat[:, :, 2] = self.B
        datr[:, :, 0] = self.R
        datg[:, :, 1] = self.G
        datb[:, :, 2] = self.B

        left, right = self.unit[0], self.unit[-1]
        ax_width = 2 * axlim

        # generate a figure and axes to plot in
        fig, ax = share_fig_ax(fig, ax)
        axr, axg, axb = make_rgb_axes(ax)

        ax.imshow(correct_gamma(dat),
                  extent=[left, right, left, right],
                  interpolation=interp_method,
                  origin='lower')

        axr.imshow(correct_gamma(datr),
                   extent=[left, right, left, right],
                   interpolation=interp_method,
                   origin='lower')
        axg.imshow(correct_gamma(datg),
                   extent=[left, right, left, right],
                   interpolation=interp_method,
                   origin='lower')
        axb.imshow(correct_gamma(datb),
                   extent=[left, right, left, right],
                   interpolation=interp_method,
                   origin='lower')

        for axs in (ax, axr, axg, axb):
            ax.set(xlim=(-axlim, axlim), ylim=(-axlim, axlim))
            if pix_grid is not None:
                # if pixel grid is desired, add it
                mult = np.floor(axlim / pix_grid)
                gmin, gmax = -mult * pix_grid, mult*pix_grid
                pts = np.arange(gmin, gmax, pix_grid)
                ax.set_yticks(pts, minor=True)
                ax.set_xticks(pts, minor=True)
                ax.yaxis.grid(True, which='minor')
                ax.xaxis.grid(True, which='minor')
        ax.set(xlabel=r'Image Plane X [$\mu m$]', ylabel=r'Image Plane Y [$\mu m$]')
        axr.text(-axlim+0.1*ax_width, axlim-0.2*ax_width, 'R', color='white')
        axg.text(-axlim+0.1*ax_width, axlim-0.2*ax_width, 'G', color='white')
        axb.text(-axlim+0.1*ax_width, axlim-0.2*ax_width, 'B', color='white')
        return fig, ax

def convpsf(psf1, psf2):
    '''Convolves two PSFs.

    Args:
        psf1 (prysm.PSF): first PSF.

        psf2 (prysm.PSF): second PSF.

    Returns:
        PSF: A new `PSF` that is the convolution of psf1 and psf2.

    Notes:
        The PSF with the lower nyquist frequency defines the sampling of the
            output.  The PSF with a higher nyquist will be truncated in the
            frequency domain (without aliasing) and projected onto the
            sampling grid of the PSF with a lower nyquist.

    '''
    if ( psf2.samples_x == psf1.samples_x and
         psf2.samples_y == psf1.samples_y and 
         psf2.sample_spacing == psf1.sample_spacing ):
        # no need to interpolate, use FFTs to convolve
        psf3 = PSF(data=abs(ifftshift(ifft2(fft2(psf1.data) * fft2(psf2.data)))),
                   samples_x=psf1.samples_x,
                   samples_y=psf1.samples_y,
                   sample_spacing=psf1.sample_spacing)
        return psf3._renorm()
    else:
        # need to interpolate, suppress all frequency content above nyquist for the less sampled psf
        if psf1.sample_spacing > psf2.sample_spacing:
            # psf1 has the lower nyquist, resample psf2 in the fourier domain to match psf1
            return _unequal_spacing_conv_core(psf1, psf2)
        else:
            # psf2 has lower nyquist, resample psf1 in the fourier domain to match psf2
            return _unequal_spacing_conv_core(psf2, psf1)

def _unequal_spacing_conv_core(psf1, psf2):
    '''Interpolates psf2 before using fft-based convolution

    Args:
        psf1 (prysm.PSF): PSF.  This one defines the sampling of the output.

        psf2 (prysm.PSF): PSF.  This one will have its frequency response
            truncated.

    Returns:
        PSF: a new `PSF` that is the convolution of psf1 and psf2.

    '''
    # map psf1 into the fourier domain
    ft1 = fft2(fftshift(psf1.data))
    unit1x = forward_ft_unit(psf1.sample_spacing, psf1.samples_x)
    unit1y = forward_ft_unit(psf1.sample_spacing, psf1.samples_y)
    # map psf2 into the fourier domain
    ft2 = fft2(fftshift(psf2.data))
    unit2x = forward_ft_unit(psf2.sample_spacing, psf2.samples_x)
    unit2y = forward_ft_unit(psf2.sample_spacing, psf2.samples_y)
    ft3 = ifftshift(resample_2d_complex(fftshift(ft2), (unit2y, unit2x), (unit1y, unit1x)))
    psf3 = PSF(data=abs(ifftshift(ifft2(ft1 * ft3))),
               sample_spacing=psf1.sample_spacing)
    return psf3._renorm()
