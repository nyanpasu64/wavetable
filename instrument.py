from numbers import Number
from typing import List

import numpy as np

from wavetable.wave_util import AttrDict, freq2pitch


def S(a, sep=' '):
    return sep.join(str(x) for x in a)



class MergeInstr:  # (_Instr):
    def __init__(self, waveseq: List[np.array], freq: Number = 1, amp: Number = 1,
                 volseq=None):
        self.waveseq = waveseq
        self.freq = freq
        self.amp = amp
        self.volseq = volseq


class Instr:
    def __init__(self, waveseq: List[np.array], cfg: dict = None):
        cfg = AttrDict(cfg)
        self.waveseq = waveseq
        self.wave_inds = cfg.get('wave_inds', None)
        self.vols = cfg.get('vols', None)
        self.freqs = cfg.get('freqs', None)

    def print_waves(self):
        strs = [S(wave) for wave in self.waveseq]
        print(';\n'.join(strs))
        print()

        wave_inds = self.wave_inds
        if wave_inds is None:
            wave_inds = list(range(len(self.waveseq)))
        print(S(wave_inds))
        print()

    def print_freqs(self, note, tranpose_factor=1):
        pitch = [freq2pitch(freq * tranpose_factor, note) for freq in self.freqs]
        print('pitch')
        print(S(pitch))
        print()

    def print(self, *args):
        self.print_waves()

        if self.vols is not None:
            print('vols:')
            print(S(self.vols))
            print()

        if self.freqs is not None:
            self.print_freqs(*args)



# def _get(seq, i):
#     """
#     >>> arr = [0,1,2]
#     >>> for i in range(10):
#     ...     assert _get(arr, i) == min(i, 2)
#
#     >>> arr = np.array(arr)
#     >>> for i in range(10):
#     ...     assert _get(arr, i) == min(i, 2)
#
#     :param waveseq:
#     :param i:
#     :return:
#     """
#     if i >= len(seq):
#         return seq[-1]
#     return seq[i]
