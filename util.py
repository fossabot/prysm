def pupil_sample_to_psf_sample(pupil_sample, num_samples, wavelength, efl):
    return (wavelength * efl * 1e3) / (pupil_sample * num_samples)

def psf_sample_to_pupil_sample(psf_sample, num_samples, wavelength, efl):
    return (psf_sample * num_samples) / (wavelength * efl * 1e3)

def correct_gamma(img):
  return img**(1/2.2)
