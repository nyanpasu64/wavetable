from numbers import Number
from typing import List, Sequence, Union

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
    seqs = ['wave_inds', 'vols', 'freqs']

    def __init__(self, waveseq: List[np.array], cfg: dict = None):
        cfg = AttrDict(cfg)
        self.waveseq = waveseq

        self.wave_inds = cfg.get('wave_inds', None)
        if self.wave_inds is None:
            self.wave_inds = np.arange(len(self.waveseq))
        self.vols = cfg.get('vols', None)
        self.freqs = cfg.get('freqs', None)

        for seq_key in self.seqs:
            seq = getattr(self, seq_key)
            if isinstance(seq, list):
                setattr(self, seq_key, np.array(seq))

    def print_waves(self):
        strs = [S(wave) for wave in self.waveseq]
        print(';\n'.join(strs))
        print()

        wave_inds = self.wave_inds
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

    def __getitem__(self, get):
        inds = []
        loop_pos = -1

        for i, elem in enumerate(get):
            if isinstance(elem, str):
                if elem == '|':
                    loop_pos = len(inds)
                else:
                    raise ValueError('cannot get item "%s"' % elem)
            else:
                inds.extend(np.r_[elem])

        end = max(inds) + 1

        sub_instr = Instr(self.waveseq[:end])
        for seq_key in self.seqs:
            seq = getattr(self, seq_key)
            if seq is not None:
                seq = seq[inds]
            setattr(sub_instr, seq_key, seq)

        return sub_instr


def deduplicate(instr: Instr):
    return instr    # todo


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
