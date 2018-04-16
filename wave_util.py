import math
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


# Rescaler

def quantize(a, y=None):
    if y is None:
        y = math.ceil(max(a))
    return np.minimum(a.astype(int), y - 1)


def iround(a):
    return np.round(a).astype(int)

class Rescaler:
    def __init__(self, maxrange: int, rounding='quantize', translate=True):
        self.max_range = maxrange
        self.rounding = rounding
        self.translate = translate

    # def __call__(self, ys):
    #     return self.rescale_quantize(ys, ret_tuple)

    def rescale_peak(self, ys: np.ndarray):
        """
        :param ys: waveform
        :return: (rescaled waveform, peak amplitude)
        """
        max_range = self.max_range

        if self.rounding == 'round':
            def _quantize(ys, _):
                return iround(ys)

            max_range -= 1

        elif self.rounding == 'quantize':
            _quantize = quantize
        elif self.rounding == 'skip':
            def _quantize(ys, _):
                return ys
        else:
            raise ValueError('self.do_round')

        if self.translate:
            ys -= np.amin(ys)
        peak = np.amax(ys)
        ys /= peak
        ys *= max_range

        out = _quantize(ys, max_range)
        return out, peak

    def rescale(self, ys):
        return self.rescale_peak(ys)[0]


def pitch2freq(pitch: int):
    freq = 440 * 2 ** ((pitch - 69) / 12)
    return freq


TICKS_PER_SEMITONE = 32


def freq2pitch(freq: float, reference: int):
    float_pitch = 12 * (np.log(freq / 440) / np.log(2)) + 69 - reference
    linear_pitch = int(round(float_pitch * TICKS_PER_SEMITONE))
    return linear_pitch


def freq2note_pitch(freq: float):
    float_pitch = 12 * (np.log(freq / 440) / np.log(2)) + 69
    note = int(round(float_pitch))
    dpitch = float_pitch - note
    linear_pitch = int(round(dpitch * TICKS_PER_SEMITONE))

    return note, linear_pitch
