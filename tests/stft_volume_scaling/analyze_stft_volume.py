from functools import wraps
from pathlib import Path
from typing import List
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy.stats import gmean

from wavetable import to_brr
from wavetable.to_brr import yaml
from wavetable.wave_reader import WaveReader, File


"""
All .wtcfg files have:
- no per-wave volume
- no mode (defaults to stft)

Using to_brr.WavetableConfig helps avoid skewing results.
- It disables ZOH FFT and all forms of scaling.

All waves are sampled with a large nsamp and nwave,
to accurately reflect the input signal.

For wave x, define magnitude |x| = RMS amplitude.
- Each file produces a handful (1 or more) of waves.
- For each file, we compute (mean |output wave|/|input segment|).
    - Interestingly, "average of ratios" is consistent among width_ms,
      while "gmean of ratios" varies with width_ms.
- Then take the geometric mean of all files.

# Results:

width_ms = None (default of 1000/30)
chunky  0.5736038885625943
intro-bass  0.6133723226848457
melodic  0.5604460452449523
sine  0.6119300097927577
square  0.6122228717284739
strings  0.6046736341320257

GMean:  0.5956626934300164

I picked File.VOLUME_RATIO = 0.585 as a compromise between
complex waves (chunky/melodic) and simple waves.
"""


def main():
    # Disable volume ratio compensation.
    File.VOLUME_RATIO = 1

    # ratios = []
    cfg_ratio = {}

    for width_ms in list(range(15, 40+1, 5)) + [None]:
        print('width_ms =', width_ms)
        for cfg_path in sorted(Path().glob('*.wtcfg')):
            # print(cfg_path)
            file_cfg: dict = yaml.load(cfg_path)
            cfg = to_brr.WavetableConfig(**file_cfg)
            if width_ms:
                cfg.width_ms = width_ms

            wr = WaveReader(cfg_path.parent, cfg)
            cfg_ratio[cfg_path] = calc_ratio(wr)
            # ratios.append(calc_ratio(wr))

        ratios = list(cfg_ratio.values())

        if width_ms is None:
            for k, v in cfg_ratio.items():
                print(k.stem, '', v)
            print()

        # print('Mean: ', np.mean(ratios))
        # print('Median: ', np.median(ratios))
        print('GMean: ', gmean(ratios))
        print()


def magnitude(wave) -> float:
    """ Compute RMS average of wave. """
    assert isinstance(wave, np.ndarray)

    wave = wave.astype(float)
    rms = np.sqrt(np.mean(wave ** 2))
    return float(rms)


# noinspection PyProtectedMember
def calc_ratio(wr: WaveReader) -> float:
    assert len(wr.files) == 1
    file = wr.files[0]

    # Hook the file to save input data.
    # [idx][time] = value
    list_data: List[np.ndarray] = []

    def save_output(_channel_data_at):
        @wraps(_channel_data_at)
        def wrapper(*args, **kwargs):
            channel_data: np.ndarray = _channel_data_at(*args, **kwargs)

            assert len(channel_data) == 1
            data = channel_data[0].copy()
            list_data.append(data)

            return channel_data
        return wrapper

    file._channel_data_at = save_output(file._channel_data_at)

    # Read output data.
    instr = wr.read()
    waves = instr.waves

    assert len(waves) == len(list_data)

    # Compare magnitude of input and output data.
    wave_ratios = []
    for input, output in zip(list_data, waves):
        wave_ratios.append(magnitude(output) / magnitude(input))

    # Using
    return float(np.mean(wave_ratios))


main()
