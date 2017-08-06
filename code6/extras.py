'''
Base meta-class for aberration simulations that provides wavefront, PSF, and MTF simulation methods
'''
from code6.util import share_fig_ax

def plot_fourier_chain(pupil, psf, mtf, fig=None, axs=None, sizex=12, sizey=6):
    '''Plots a pupil, PSF, and MTF demonstrating the process of fourier optics simulation

    Args:
        pupil (code6.Pupil): The pupil of an optical system
        psf (code6.PSF): The psf of an optical system
        mtf (code6.MTF): The MTF of an optical system
        fig (pyplot.figure): A figure object
        axs (list of pyplot.axes): axes to place the plots in
        sizex (number): size of the figure in x
        sizey (number): size of the figure in y

    Returns:
        fig, axs.  A figure and axes containing the plot.
    '''

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
