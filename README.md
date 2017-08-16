**prysm** is an optical modeling and analysis toolkit.  It performs:

* Pupil modeling via:

* * Seidel notation

* * Fringe Zernike notation

* * Noll / Standard Zernike notation

* and with:

* * gaussian apodization

* * noncircular apertures 

* * * n-sided polygons

* * * rotated ellipses

* * * user-provided masks

* Thin lens / geometrical optics modeling

* PSF simulation

* MTF simulation

* Detector sampling

* Image synthesis

These capabilities can be used for, e.g.:

* aberration modeling at any point in the imaging chain

* through-focus simulation of PSF/MTF in image or object space coordinates

* phase retrieval

**prysm** depends heavily on the rich ecosystem of numerical and scientific python libraries, esp. numpy and scipy.  It is highly recommended to use Anaconda python, or another distribution that comes with precompiled version of numpy and scipy that use BLAS or LLVM to accelerate array operations.

**prysm** is made freely available and distributed under an MIT license.  It is provided without warranty.  The author may be reached for support via [email](mailto:brandondube@gmail.com).
