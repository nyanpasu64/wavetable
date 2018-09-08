from pathlib import Path

import numpy as np

from wavetable.wave import load_wave


ROOT = Path('tests/test_waves')

MONO_PATH = ROOT / 'Sample 19.wav'
STEREO_PATH = ROOT / 'stereo out of phase.wav'


def test_wave_mono():
    sr, data = load_wave(str(MONO_PATH))     # type: int, np.ndarray

    # Assert explicit mono channel
    assert data.ndim == 2
    assert data.shape[1] == 1


def test_wave_stereo():
    sr, data = load_wave(str(STEREO_PATH))     # type: int, np.ndarray

    # Assert explicit mono channel
    assert data.ndim == 2
    assert data.shape[1] == 2


