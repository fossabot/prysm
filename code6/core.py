'''
Base meta-class for aberration simulations that provides wavefront, PSF, and MTF simulation methods
'''
import numpy as np
from numpy import arctan2, exp, floor, nan, sqrt, pi
from numpy import power as npow
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from matplotlib import pyplot as plt

from code6.lens import Lens
from code6.fringezernike import FringeZernike
from code6.psf import PSF
from code6.otf import MTF
from code6.fttools import pad2d
from code6.util import pupil_sample_to_psf_sample, psf_sample_to_pupil_sample, share_fig_ax

def plot_fourier_chain(pupil, psf, mtf, fig=None, axs=None, sizex=12, sizey=6):

    # create a figure
    fig, axs = share_fig_ax(fig, axs, numax=3)

    pupil.plot2d(fig=fig, ax=axs[0])
    psf.plot2d(fig=fig, ax=axs[1])
    mtf.plot2d(fig=fig, ax=axs[2])

    axs[0].set(title='Pupil')
    axs[1].set(title='PSF')
    axs[2].set(title='MTF')

    bbox_props = dict(boxstyle="rarrow", fill=None, lw=1)
    axs[0].text(1.75, 1.25, r'|Fourier Transform|$^2$', ha='center', va='center', bbox=bbox_props)
    axs[0].text(5.3, 1.25, r'|Fourier Transform|', ha='center', va='center', bbox=bbox_props)
    fig.set_size_inches(sizex, sizey)
    fig.tight_layout()
    return fig, axs

def convolve_pixel(pitch, psf):
    error = psf.sample_spacing/2
    print(f'This function does not resample, pixel pitch is enlarged by {error}')
    half_width = pitch / 2
    steps = int(np.floor(half_width / psf.sample_spacing))
    pixel_aperture = np.zeros((psf.samples, psf.samples))
    pixel_aperture[psf.center-steps:psf.center+steps, psf.center-steps:psf.center+steps] = 1
    ft_ape = fft2(pixel_aperture)
    ft_psf = fft2(psf.data)
    return PSF(data=ifftshift(ifft2(np.absolute(ft_ape * ft_psf))),
               samples=psf.samples,
               sample_spacing = psf.sample_spacing)
