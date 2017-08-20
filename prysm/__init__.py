''' prysm -- a python optics module
'''

from prysm.extras import plot_fourier_chain
from prysm.detector import Detector, OLPF, PixelAperture
from prysm.pupil import Pupil
from prysm.fringezernike import FringeZernike
from prysm.standardzernike import StandardZernike
from prysm.seidel import Seidel
from prysm.surfacefinish import SurfaceFinish
from prysm.psf import PSF, MultispectralPSF, RGBPSF
from prysm.otf import MTF
from prysm.geometry import ( gaussian, rotated_ellipse, triangle, square,
                             pentagon, hexagon, heptagon, octagon, nonagon,
                             decagon, hendecagon, dodecagon, trisdecagon )
from prysm.objects import Image, RGBImage, Slit, Pinhole, SiemensStar

__all__ = [
    'plot_fourier_chain',
    'Detector',
    'OLPF',
    'PixelAperture',
    'Pupil',
    'FringeZernike',
    'StandardZernike',
    'Seidel',
    'SurfaceFinish',
    'PSF',
    'MultispectralPSF',
    'RGBPSF',
    'MTF',
    'gaussian',
    'rotated_ellipse',
    'triangle',
    'square',
    'pentagon',
    'hexagon',
    'heptagon',
    'octagon',
    'nonagon',
    'decagon',
    'hendecagon',
    'dodecagon',
    'trisdecagon',
    'Image',
    'RGBImage',
    'Slit',
    'Pinhole',
    'SiemensStar',
]
