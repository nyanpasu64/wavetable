import numpy as np
from numpy.fft import ifft, fft


class AttrDict(dict):
    def __init__(self, seq=None, **kwargs):
        if seq is None:
            seq = {}
        super(self.__class__, self).__init__(seq, **kwargs)
        self.__dict__ = self


# Phasor merging


def _power_sum(arr):
    square = np.array(arr) ** 2
    mean = np.sum(square)
    root = np.sqrt(mean)
    return root


def power_merge(phasors):
    mag = _power_sum(np.abs(phasors))
    arg = np.angle(np.sum(phasors))
    return mag * np.exp(1j * arg)


def amplitude_merge(phasors):
    mag = np.sum(np.abs(phasors))
    arg = np.angle(np.sum(phasors))
    return mag * np.exp(1j * arg)


sum_merge = np.sum


# Correlation

def correlate(fixed, sweep):
    """ circular cross-correlation of 2 equal waves """
    fixed = np.array(fixed)
    sweep = np.array(sweep)
    if fixed.shape != sweep.shape or len(fixed.shape) != 1:
        raise ValueError('incorrect dimensions: %s versus %s' % (fixed.shape, sweep.shape))

    return ifft(fft(fixed) * fft(sweep).conj()).real


def correlate_offset(fixed, sweep, i=None):
    """ Get peak correlation offset. """

    corrs = correlate(fixed, sweep)
    if np.argmax(abs(corrs)) != np.argmax(corrs):
        raise ValueError(f'yeah, seems like you need to invert wave {i}')

    offset = np.argmax(corrs)
    return offset


def align_waves(waveseq):
    """ Returns maximum-correlation copy of waveseq. """
    out = [waveseq[0]]
    for i, wave in enumerate(waveseq[1:], 1):
        offset = correlate_offset(out[-1], wave, i)
        out.append(np.roll(wave, offset))

    return out
