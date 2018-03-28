from numbers import Number
from typing import List

import numpy as np

from wave_util import AttrDict


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
