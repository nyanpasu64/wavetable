import warnings
from enum import IntEnum
from typing import Tuple

import numpy as np
from scipy.io import wavfile


class StereoMode(IntEnum):
    ALL = 1
    LEFT = 2

    @classmethod
    def parse(cls, value) -> 'StereoMode':
        try:
            if not isinstance(value, cls):
                value = cls[value]
        except KeyError:
            # noinspection PyUnresolvedReferences
            raise ValueError(
                f'invalid {cls.name} {value} not in '
                f'{[el.name for el in cls]}')
        return value


# FIXME rename to read_wave
def load_wave(wav_path: str, stereo=StereoMode.ALL) -> Tuple[int, np.ndarray]:
    """Loads wave from file. Optionally merges data. Returns sr, data[index][chan]. """

    # Polyphone SF2 wavs contain 'smpl' chunk with loop data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(wav_path)  # type: int, np.ndarray

    # Cast to data[index][chan]
    if data.ndim == 1:
        data = data.reshape(len(data), 1)

    # Cast to desired stereo layout
    stereo = StereoMode.parse(stereo)
    if stereo == StereoMode.ALL:
        pass
    elif stereo == StereoMode.LEFT:
        data = data[:, [0]]
    else:
        raise ValueError(f"Wave cannot handle StereoMode={stereo}")

    return sr, data.astype(np.float32)   # TODO divide by peak


