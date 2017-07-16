import numpy as np
from matplotlib import pyplot as plt

def pupil_sample_to_psf_sample(pupil_sample, num_samples, wavelength, efl):
    return (wavelength * efl * 1e3) / (pupil_sample * num_samples)

def psf_sample_to_pupil_sample(psf_sample, num_samples, wavelength, efl):
    return (psf_sample * num_samples) / (wavelength * efl * 1e3)

def correct_gamma(img, encoding=2.2):
    return np.power(img, (1/encoding))

def share_fig_ax(fig=None, ax=None, numax=1):
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=numax, dpi=100)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    return fig, ax