import numpy as np
from wavetable.merge import Instr


def filtered_saw_waves(N):
    saw = np.linspace(-1, 1, 32)
    # saw = 4 ** np.linspace(0, 1, 32)
    filtered = saw + np.roll(saw, 1)

    waves = []
    for i in range(N):
        wave = np.sum([
            filtered,
            np.roll(filtered, i + 7) * 0.8,
            np.roll(filtered, 2 * i) * 0.4,
        ], 0)
        waves.append(wave)  # .astype(int)

    return waves


def saw_waves(N):
    saw = np.linspace(-1, 1, 32)

    waves = []
    for i in range(N):
        wave = np.sum([
            saw,
            np.roll(saw, i + 7) * 0.8,
            np.roll(saw, 2 * i) * 0.4,
        ], 0)
        waves.append(wave)  # .astype(int)

    return waves


