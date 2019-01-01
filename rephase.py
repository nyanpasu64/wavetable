"""
Not imported by any other file, but somewhat useful on its own.
"""
import numpy as np
from wavetable.dsp.wave_util import Rescaler
from wavetable.instrument import MML


def rephase(wave, phase_f, range):
    r = Rescaler(range)

    wave = MML(wave)

    avg_fft = np.fft.rfft(wave)
    first_bin = 1

    phases = phase_f(np.arange(first_bin, len(avg_fft)))
    phasors = np.exp(1j * phases)

    avg_fft = np.abs(avg_fft).astype(complex)
    avg_fft[first_bin:] *= phasors

    out = np.fft.irfft(avg_fft)
    out = r.rescale(out).view(MML)
    return str(out)
