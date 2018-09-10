import math
from typing import NamedTuple

import numpy as np
from numpy.fft import ifft, fft


def A(*args):
    return np.array(args)


# Phasor merging
# TODO dsp.py


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
        raise ValueError(
            'incorrect dimensions: %s versus %s' % (fixed.shape, sweep.shape))

    return ifft(fft(fixed) * fft(sweep).conj()).real


class Correlation(NamedTuple):
    offset: int
    should_invert: bool


def correlate_offset(fixed, sweep) -> Correlation:
    """ Get peak correlation offset. """
    # TODO how to handle rounded values? You'd need to know maxval.
    corrs = correlate(fixed, sweep)

    offset = np.argmax(abs(corrs))      # type: int
    should_invert = (corrs[offset] < 0)
    return Correlation(offset, should_invert)


def align_waves(waves):
    """ Returns maximum-correlation copy of waves. """
    out = [waves[0]]
    for i, wave in enumerate(waves[1:], 1):
        state = correlate_offset(out[-1], wave)

        correlated = np.roll(wave, state.offset)
        if state.should_invert:
            correlated *= -1

        out.append(correlated)

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

    def rescale_peak(self, ys):
        """
        :param ys: waveform
        :return: (rescaled waveform, peak amplitude)
        """
        ys = np.asarray(ys)
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


# Pitch conversion


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
