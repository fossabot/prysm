''' Extra functions, used for demos and education
'''
from prysm.util import share_fig_ax


def plot_fourier_chain(pupil, psf, mtf, fig=None, axs=None, sizex=12, sizey=6):
    '''Plots a :class:`Pupil`, :class:`PSF`, and :class:`MTF`
        demonstrating the process of fourier optics simulation

    Args:
        pupil (prysm.Pupil): The pupil of an optical system

        psf (prysm.PSF): The psf of an optical system

        mtf (prysm.MTF): The MTF of an optical system

        fig (pyplot.figure): A figure object

        axs (list of pyplot.axes): axes to place the plots in

        sizex (number): size of the figure in x

        sizey (number): size of the figure in y

    Returns:
        fig, axs.  A figure and axes containing the plot.

    '''

    fig, axs = share_fig_ax(fig, axs, numax=3)

    pupil.interferogram(fig=fig, ax=axs[0])
    psf.plot2d(fig=fig, ax=axs[1])
    mtf.plot2d(fig=fig, ax=axs[2])

    axs[0].set(title='Pupil')
    axs[1].set(title='PSF')
    axs[2].set(title='MTF')

    bbox_props = dict(boxstyle="rarrow", fill=None, lw=1)
    axs[0].text(1.385, 1.07, r'|Fourier Transform|$^2$',
                ha='center', va='center', bbox=bbox_props,
                transform=axs[0].transAxes)
    axs[0].text(3.15, 1.07, r'|Fourier Transform|',
                ha='center', va='center', bbox=bbox_props,
                transform=axs[0].transAxes)
    fig.set_size_inches(sizex, sizey)
    fig.tight_layout()
    return fig, axs
