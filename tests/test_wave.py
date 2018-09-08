from pathlib import Path

from wavetable.wave_reader import Wave


WAV_PATH = Path('test_waves/Sample 19.wav')


def test_wave():
    Wave(str(WAV_PATH))
