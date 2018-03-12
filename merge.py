from typing import List
import itertools
import numpy as np
from collections import namedtuple, OrderedDict
from global_util import *

from . import fourier
from . import gauss
from .gauss import set_range
assert set_range



def _rms(ar):
    square = np.array(ar) ** 2
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root

def load_string(s):
    data = s.split(';')
    waves = []
    for d in data:
        wave = I(d)
        if wave.size > 0:
            waves.append(wave)
    return np.array(waves)


def _load_waves(filename):
    with open(filename) as f:
        s = f.read()
    return load_string(s)


# **** merge


def _merge_with(*waves: List[np.ndarray], nsamp, avg_func):
    ffts = [np.fft.rfft(wave) for wave in waves]
    
    outs = []
    for coeffs in itertools.zip_longest(*ffts, fillvalue=0j):
        mag = _rms(np.abs(coeffs))
        arg = np.mean(np.angle(coeffs))
        outs.append(mag * np.exp(1j * arg))
    
    wave_out = fourier.irfft(outs, nsamp)
    return gauss.rescale_quantize(wave_out)


def amp_merge(*waves: List[np.ndarray], nsamp=None):
    """ Combines multiple waves[], taking the average *amplitude* of each harmonic. """
    return _merge_with(*waves, nsamp=nsamp, avg_func=np.mean)

def power_merge(*waves: List[np.ndarray], nsamp=None):
    """ Combines multiple waves[], taking the average *power* of each harmonic. """
    return _merge_with(*waves, nsamp=nsamp, avg_func=_rms)


# **** packing MML strings

def load_file_mml(filename, mml):
    waves = _load_waves(filename)
    inds = I(mml)
    return waves[inds]


Instr = namedtuple('Instr', 'waveseq freq amp')
# Partial = namedtuple('Instr', 'wave freq amp')

def waveseq_get(waveseq, i):
    if i >= len(waveseq):
        return waveseq[-1]
    return waveseq[i]


def _merge_harmonic(instrs: List[Instr], wave_num, nsamp):
    harmonic_waves = []

    # entry[i] = [freq, amp]
    for instr in instrs:
        wave = waveseq_get(instr.waveseq, wave_num)
        harmonic_wave = npcat([wave] * instr.freq) * instr.amp
        harmonic_waves.append(harmonic_wave)

    out = power_merge(*harmonic_waves, nsamp=nsamp)
    return out


def _pad_waves(waves, length):
    return cat(waves, [waves[-1]] * (length - len(waves)))


def merge_instrs(instrs: List[Instr], nsamp):
    """ Pads each Instr to longest. Then merges all and returns new $waves. """
    length = max(len(instr.waveseq) for instr in instrs)
    merged_waveseq = []

    for i in range(length):
        merged_waveseq.append(_merge_harmonic(instrs, i, nsamp))

    return merged_waveseq

def combine(waveseq):
    """ Returns minimal waveseq, MML string. """
    waveseq = [tuple(wave) for wave in waveseq]
    wave2idx = OrderedDict()
    curr_idx = 0

    mml = []

    for wave in waveseq:
        if wave not in wave2idx:
            wave2idx[wave] = curr_idx
            curr_idx += 1
        mml.append(wave2idx[wave])

    minimal_waveseq = [S(wave) for wave in wave2idx.keys()]
    print(';\n'.join(minimal_waveseq))
    print()
    print(S(mml))


def merge_combine(instrs: List[Instr], nsamp):
    """ merge and combine into minimal wave and MML string. """
    combine(merge_instrs(instrs, nsamp))
