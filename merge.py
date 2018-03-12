from typing import List
import itertools
import numpy as np

from . import fourier
from . import gauss
from .gauss import set_range
assert set_range



def _I(s, *args, **kwargs):
    return np.array([int(x, *args, **kwargs) for x in s.split()])


def _rms(ar):
    square = np.array(ar) ** 2
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root


def load_waves(filename):
    with open(filename) as f:
        data = f.read().split(';')
    waves = []
    for d in data:
        wave = _I(d)
        if wave.size > 0:
            waves.append(wave)
    return np.array(waves)


# **** merge


def _merge_with(*waves: List[np.ndarray], nsamp, avg_func):
    ffts = [np.fft.rfft(wave) for wave in waves]
    
    outs = []
    for coeffs in itertools.zip_longest(*ffts, fillvalue=0j):
        mag = _rms(np.abs(coeffs))
        arg = np.mean(np.angle(coeffs))
        outs.append(mag * np.exp(1j * arg))
    
    wave_out = fourier.irfft(outs)
    return gauss.rescale_quantize(wave_out)


def merge(*waves: List[np.ndarray], nsamp=None):
    """ Combines multiple waves[], taking the average *amplitude* of each harmonic. """
    return _merge_with(*waves, nsamp=nsamp, avg_func=np.mean)

def power_merge(*waves: List[np.ndarray], nsamp=None):
    """ Combines multiple waves[], taking the average *power* of each harmonic. """
    return _merge_with(*waves, nsamp=nsamp, avg_func=_rms)


