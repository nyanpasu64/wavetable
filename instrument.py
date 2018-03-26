from numbers import Number
from typing import List

import numpy as np


class Instr:  # (_Instr):
    def __init__(self, waveseq: List[np.array], freq: Number=1, amp: Number=1,
                 volseq=None):
        self.waveseq = waveseq
        self.freq = freq
        self.amp = amp
        self.volseq = volseq
