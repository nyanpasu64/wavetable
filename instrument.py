from numbers import Number
from typing import List, Sequence, Union

import numpy as np
from wavetable.wave_util import AttrDict, freq2pitch

HEX = ['0x', '$']
LOOP = '|'
RELEASE = '/'


class MML(np.ndarray):
    """
    >>> m = MML('0 1 2 3 4 | 5 5 5')
    >>> print(m)
    0 1 2 3 4 | 5 5 5
    >>> print(m[1::2])
    1 3 | 5 5
    """

    def __new__(cls, string):
        arr = []
        loop = None
        release = None

        a = string.split()
        for word in a:
            word0 = word
            if word == LOOP:
                loop = len(arr)
                continue
            elif word == RELEASE:
                release = len(arr)
                continue

            base = 10
            for hex in HEX:
                if word.startswith(hex):
                    base = 16
                    word = word[len(hex):]

            try:
                arr.append(int(word, base))
            except ValueError:
                print('warning invalid word', word0)

        new = np.asarray(arr).view(cls)
        new.loop = loop
        new.release = release

        return new

    def __array_finalize__(self, old):
        if old is None:
            return
        self.loop = getattr(old, 'loop', None)
        self.release = getattr(old, 'release', None)

    def __str__(self):
        out = []
        for i, el in enumerate(self):
            if i == self.loop:
                out.append(LOOP)
            if i == self.release:
                out.append(RELEASE)
            out.append(str(el))
        return ' '.join(out)

    def __repr__(self):
        return f"{type(self).__name__}('{str(self)}')"

    def __getitem__(self, key):
        new = np.ndarray.__getitem__(self, key).view(type(self))
        if isinstance(key, slice):
            start = key.start or 0
            step = key.step or 1

            def munge(idx):
                if idx is None:
                    return None
                return (idx - start - 1) // step + 1

            new.loop = munge(new.loop)
            new.release = munge(new.release)
        return new

    def __truediv__(self, other):
        return round(np.ndarray.__truediv__(self, other))

    def __round__(self):
        # out = np.round(self)
        out = np.ceil(self - 0.5)
        return out.astype(int)


def I(s, *args, **kwargs):
    return np.array([int(x, *args, **kwargs) for x in s.split()])

def F(s):
    return np.array([float(x) for x in s.split()])

def S(a, sep=' '):
    if isinstance(a, MML):
        return str(a)
    else:
        return sep.join(str(x) for x in a)


class Instr:
    seqs = ['wave_inds', 'vols', 'freqs']

    def __init__(self, waveseq: List[np.array], cfg: dict = None, **kwargs):
        cfg = AttrDict(cfg, **kwargs)
        self.waveseq = [np.asarray(wave) for wave in waveseq]

        self.wave_inds = cfg.get('wave_inds', None)
        if self.wave_inds is None:
            self.wave_inds = np.arange(len(self.waveseq))
        self.vols = cfg.get('vols', None)
        self.freqs = cfg.get('freqs', None)

        for seq_key in self.seqs:
            seq = getattr(self, seq_key)
            if seq is not None:
                setattr(self, seq_key, np.array(seq).view(MML))

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
        loop_pos = None
        release_pos = None

        for i, elem in enumerate(get):
            if isinstance(elem, str):
                if elem == LOOP:
                    loop_pos = len(inds)
                elif elem == RELEASE:
                    release_pos = len(inds)
                else:
                    raise ValueError('cannot get item "%s"' % elem)
            else:
                inds.extend(np.r_[elem])

        end = max(inds) + 1

        sub_instr = Instr(self.waveseq[:end])
        for seq_key in self.seqs:
            seq = getattr(self, seq_key)
            if seq is not None:
                seq = seq[inds]  # .view(MML) # type: MML
                seq.loop = loop_pos
                seq.release = release_pos
            setattr(sub_instr, seq_key, seq)

        return sub_instr


def deduplicate(instr: Instr):
    return instr  # todo


def _get(waveseq: Sequence, i: int):
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


def _normalize(numeric_arg: Union[Number, Sequence, None]) -> Sequence[Number]:
    if numeric_arg is None:
        return [1]
    if isinstance(numeric_arg, Number):
        return [numeric_arg]
    return numeric_arg


class MergeInstr(Instr):
    def __init__(self, waveseq: List[np.array], harmonic: int = 1,
                 vols: Union[Number, Sequence, MML] = None):

        # super().'wave_inds', 'vols', 'freqs'
        # We only use vols.

        super().__init__(waveseq, vols=_normalize(vols))
        self.harmonic = harmonic

    def get_wave_scaled(self, idx):
        wave = _get(self.waveseq, idx)
        scaled = np.asarray(wave, float) * _get(self.vols, idx)
        freqd = np.concatenate([scaled] * self.harmonic)
        return freqd
