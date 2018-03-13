from typing import List
import itertools
from collections import namedtuple, OrderedDict
from global_util import *

from wavetable import fourier
from wavetable import gauss
from wavetable.gauss import set_range

assert set_range


def _rms(arr):
    square = np.array(arr) ** 2
    mean = np.mean(square)
    root = np.sqrt(mean)
    return root


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


Instr = namedtuple('Instr', 'waveseq freq amp')


class Merge:
    def _merge_waves(self, waves: List[np.ndarray], nsamp):
        """ Depends on self.avg_func. """
        ffts = [np.fft.rfft(wave) for wave in waves]

        outs = []
        for coeffs in itertools.zip_longest(*ffts, fillvalue=0j):
            mag = self.avg_func(np.abs(coeffs))
            arg = np.angle(np.mean(coeffs))
            outs.append(mag * np.exp(1j * arg))

        wave_out = fourier.irfft(outs, nsamp)
        return gauss.rescale_quantize(wave_out)

    # def amp_merge(self, *waves: List[np.ndarray], nsamp=None):
    #     """ Combines multiple waves[], taking the average *amplitude* of each harmonic. """
    #     return self._merge_with(*waves, nsamp=nsamp, avg_func=np.mean)
    #
    # def power_merge(self, *waves: List[np.ndarray], nsamp=None):
    #     """ Combines multiple waves[], taking the average *power* of each harmonic. """
    #     return self._merge_with(*waves, nsamp=nsamp, avg_func=_rms)

    def _merge_instrs_idx(self, instrs: List[Instr], wave_num, nsamp):
        # TODO: high coupling, integrate?
        harmonic_waves = []

        # entry[i] = [freq, amp]
        for instr in instrs:
            wave = _get(instr.waveseq, wave_num)
            harmonic_wave = npcat([wave] * instr.freq) * instr.amp
            harmonic_waves.append(harmonic_wave)

        out = self._merge_waves(harmonic_waves, nsamp=nsamp)
        return out

    # @staticmethod
    # def _pad_waves(waves, length):
    #     return cat(waves, [waves[-1]] * (length - len(waves)))

    def merge_instrs(self, instrs: List[Instr], nsamp):
        """ Pads each Instr to longest. Then merges all and returns new $waves. """
        length = max(len(instr.waveseq) for instr in instrs)
        merged_waveseq = []

        for i in range(length):
            merged_waveseq.append(self._merge_instrs_idx(instrs, i, nsamp))

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

        minimal_waveseq = [S(wave) for wave in wave2idx.keys()]
        print(';\n'.join(minimal_waveseq))
        print()
        print(S(mml))
        print()
        print()

    def merge_combine(self, instrs: List[Instr], nsamp):
        """ merge and combine into minimal wave and MML string. """
        self.combine(self.merge_instrs(instrs, nsamp))

    def __init__(self, avg_func=np.mean):
        self.avg_func = avg_func


def merge_combine(instrs: List[Instr], nsamp, avg_func=np.mean):
    merger = Merge(avg_func)
    merger.merge_combine(instrs, nsamp)
