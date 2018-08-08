from numbers import Number
from typing import List, Sequence, Union, ClassVar, Dict

import numpy as np
from dataclasses import dataclass, fields, Field

from wavetable.util.math import seq_along
from wavetable.wave_util import freq2pitch

HEX = ['0x', '$']
LOOP = '|'
RELEASE = '/'


def I(s, *args, **kwargs):
    return np.array([int(x, *args, **kwargs) for x in s.split()])


def F(s):
    return np.array([float(x) for x in s.split()])


def S(a, sep=' '):
    if isinstance(a, MML):
        return str(a)
    else:
        return sep.join(str(x) for x in a)


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

    def __round__(self):
        out = np.ceil(self - 0.5)
        return out.astype(int)


# TODO replace freqs with pitches (MIDI), add "root_pitch: int"

SeqType = Union[MML, np.ndarray, list, None]  # Converted to Optional[MML]


@dataclass(eq=False)
class Instr:
    waves: Union[List[np.ndarray], np.ndarray]  # Converted to 2D ndarray

    sweep: SeqType = None
    vols: SeqType = None
    freqs: SeqType = None

    SEQS: ClassVar = ['sweep', 'vols', 'freqs']

    @property
    def seqs(self) -> Dict[str, np.ndarray]:
        return {seq: getattr(self, seq) for seq in self.SEQS}

    def __post_init__(self):
        # Convert waves to ndarrays
        for wave in self.waves:
            if len(wave) != len(self.waves[0]):
                raise ValueError('Invalid Instr with unequal wave lengths')

        self.waves = np.asarray(self.waves)

        # Linear sweep through all waves
        if self.sweep is None:
            self.sweep = seq_along(self.waves)

        # Convert sequences to MML
        for k, seq in self.seqs.items():
            if seq is not None:
                setattr(self, k, np.array(seq).view(MML))

    def __eq__(self, other):
        if not isinstance(other, Instr):
            return NotImplemented

        # Ensure that all ndarrays are equal.
        field: Field
        for field in fields(self):
            name = field.name
            if getattr(self, name).tolist() != getattr(other, name).tolist():
                return False
        return True

    def __getitem__(self, get):
        """ Return the specified frames from the instrument.
        Does not remove unused waves. """
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

        # Pick frames from the sweep/volume/pitch.
        sub_instr = Instr(self.waves)
        for k, seq in self.seqs.items():
            if seq is not None:
                seq = seq[inds]
                seq.loop = loop_pos
                seq.release = release_pos
            setattr(sub_instr, k, seq)

        # TODO call sub_instr.remove_unused_waves()? it reduces testability
        return sub_instr

    def remove_unused_waves(self) -> None:
        """ Remove unused waves. Mutates self in-place. """
        used_wave_idx, used2out = np.unique(self.sweep, return_inverse=True)
        self.waves = self.waves[used_wave_idx]
        self.sweep[:] = used2out

    def print(self, *args):
        self.print_waves()

        if self.vols is not None:
            print('vols:')
            print(S(self.vols))
            print()

        if self.freqs is not None:
            self.print_freqs(*args)

    def print_waves(self):
        strs = [S(wave) for wave in self.waves]
        print(';\n'.join(strs))
        print()

        wave_inds = self.sweep
        print(S(wave_inds))
        print()

    def print_freqs(self, note, tranpose_factor=1):
        pitch = [freq2pitch(freq * tranpose_factor, note) for freq in self.freqs]
        print('pitch')
        print(S(pitch))
        print()


def _get(waves: Sequence, i: int):
    """
    >>> arr = [0,1,2]
    >>> for i in range(10):
    ...     assert _get(arr, i) == min(i, 2)

    >>> arr = np.array(arr)
    >>> for i in range(10):
    ...     assert _get(arr, i) == min(i, 2)

    :param waves:
    :param i:
    :return:
    """
    if i >= len(waves):
        return waves[-1]
    return waves[i]


def _normalize(numeric_arg: Union[Number, Sequence, None]) -> Sequence[Number]:
    if numeric_arg is None:
        return [1]
    if isinstance(numeric_arg, Number):
        return [numeric_arg]
    return numeric_arg


@dataclass
class MergeInstr(Instr):
    def __init__(self, waves: List[np.array], **kwargs: Union[Number, Sequence, MML]):
        # super().'wave_inds', 'vols', 'freqs'
        # We only use vols.

        super().__init__(waves, vols=_normalize(kwargs[vols]))
        self.harmonic = kwargs[harmonic]

    def get_wave_scaled(self, idx):
        wave = _get(self.waves, idx)
        scaled = np.asarray(wave, float) * _get(self.vols, idx)
        freqd = np.concatenate([scaled] * self.harmonic)
        return freqd
