from typing import TypeVar

import numpy as np


def ceildiv(n: int, d: int) -> int:
    return -(-n // d)


def seq_along(arr):
    return np.arange(len(arr))


# wave_reader

def nearest_sub_harmonic(precise: float, accurate: float) -> float:
    """ Finds the nearest sub/harmonic of `precise` to `accurate`. """
    if precise > accurate:
        precise /= round(precise / accurate)
    elif precise < accurate:
        if precise == 0:
            return accurate
        precise *= round(accurate / precise)
    return precise


def midi2ratio(note, cents=0):
    """ Converts semitones to a frequency ratio. """
    ratio = 2 ** ((note + cents / 100) / 12)
    return ratio


def midi2freq(note, cents=0):
    """ Converts a MIDI note to an absolute frequency (Hz). """
    freq = 440 * midi2ratio(note - 69, cents)
    return freq


Numbers = TypeVar('Numbers', float, np.ndarray)


def freq2midi(freq: Numbers) -> Numbers:
    freq_ratio = freq / 440
    semitones = 12 * (np.log(freq_ratio) / np.log(2))
    return semitones + 69
