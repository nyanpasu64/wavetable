import itertools
from collections import OrderedDict
from enum import Enum
from typing import List

from global_util import *
from wavetable import fourier
from wavetable import transfer
from wavetable.instrument import MergeInstr
from wavetable.wave_util import Rescaler

import wave_util


def load_string(s):
    """
    >>> (load_string('  0 1 2 3; 4 5 6 7  ;;;  ') == [[0,1,2,3], [4,5,6,7]]).all()
    True

    :param s: a list of number-space waveforms (of equal length), separated by semicolons
    :return: an np.array[i] == one waveform.
    """

    data = s.split(';')
    waves = []
    for d in data:
        wave = I(d)
        if wave.size > 0:
            waves.append(wave)
            assert wave.shape == waves[0].shape
    return np.array(waves)


def _load_waves(filename):
    with open(filename) as f:
        s = f.read()
    return load_string(s)


def _get(waveseq, i):
    """
    >>> arr = [0,1,2]
    >>> for i in range(10):
    ...     assert _get(arr, i) == min(i, 2)

    >>> arr = np.array(arr)
    >>> for i in range(10):
    ...     assert _get(arr, i) == min(i, 2)

    :param waveseq:
    :param i:
    :return:
    """
    if i >= len(waveseq):
        return waveseq[-1]
    return waveseq[i]


# **** packing MML strings

def merge_waves_mml(waves, mml=None, vol_curve=None):
    """
    :param waves:
    :param mml:
    :param vol_curve:
    :return:
    """

    if mml:
        inds = I(mml)
        seq = waves[inds]
    else:
        seq = waves

    if vol_curve:
        vol_curve = F(vol_curve)
        cnt = max(len(seq), len(vol_curve))
        seq = np.array([
            _get(seq, i) * _get(vol_curve, i)
            for i in range(cnt)
        ])

    return seq


def load_file_mml(filename, mml=None, vol_curve=None):
    """
    >>>
    :param filename:
    :param mml:
    :param vol_curve:
    :return:
    """
    waves = _load_waves(filename)
    return merge_waves_mml(waves, mml, vol_curve)


def print_waves(waveseq):
    strs = [S(wave) for wave in waveseq]
    print(';\n'.join(strs))
    print()


print_waveseq = print_waves


class MergeStyle(Enum):
    POWER = 1
    AMPLITUDE = 2
    SUM = 3


_MAXRANGE = 16


class Merge:
    merge_funcs = {
        MergeStyle.POWER: wave_util.power_merge,
        MergeStyle.AMPLITUDE: wave_util.amplitude_merge,
        MergeStyle.SUM: wave_util.sum_merge
    }

    def __init__(self, maxrange,
                 merge_style: MergeStyle = MergeStyle.POWER,
                 scaling='local'):

        self.phasor_merger = self.merge_funcs[merge_style]
        self.scaling = scaling
        self.rescaler = Rescaler(maxrange)

    def _merge_waves(self, waves: List[np.ndarray], nsamp, transfer):
        """ Depends on self.avg_func. """
        ffts = [np.fft.rfft(wave) for wave in waves]

        outs = []
        for f, phasors in enumerate(itertools.zip_longest(*ffts, fillvalue=0j)):
            outs.append(self.phasor_merger(phasors) * transfer(f))

        wave_out = fourier.irfft(outs, nsamp)
        if self.scaling == 'local':
            return self.rescaler.rescale_quantize(wave_out)
        else:
            return wave_out

    def merge_instrs(self, instrs: List[MergeInstr], nsamp, transfer=transfer.Unity()):
        """ Pads each MergeInstr to longest. Then merges all and returns new $waves. """
        length = max(len(instr.waveseq) for instr in instrs)
        merged_waveseq = []

        for wave_idx in range(length):
            harmonic_waves = []

            # entry[wave_idx] = [freq, amp]
            for instr in instrs:
                wave = _get(instr.waveseq, wave_idx)
                harmonic_wave = npcat([wave] * instr.freq) * instr.amp
                harmonic_waves.append(harmonic_wave)

            out = self._merge_waves(harmonic_waves, nsamp=nsamp, transfer=transfer)
            merged_waveseq.append(out)

        if self.scaling == 'global':
            return self.rescaler.rescale_quantize(merged_waveseq)
        else:
            return merged_waveseq

    @staticmethod
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

        minimal_waveseq = list(wave2idx.keys())
        print_waveseq(minimal_waveseq)
        print(S(mml))
        print()
        print()

    def merge_combine(self, instrs: List[MergeInstr], nsamp, transfer=transfer.Unity()):
        """ merge and combine into minimal wave and MML string. """
        self.combine(self.merge_instrs(instrs, nsamp, transfer))


# FIXME maxrange
def merge_combine(instrs: List[MergeInstr], nsamp, avg_func=np.mean, transfer=transfer.Unity()):
    merger = Merge(avg_func)
    merger.merge_combine(instrs, nsamp, transfer)
