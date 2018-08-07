import numpy as np

from wavetable.wave_util import A, freq2midi


def test_freq2midi():
    a4 = 69
    octave = 12

    freq = A(440, 880, 220)
    midi_expected = A(a4, a4 + octave, a4 - octave)

    midi = freq2midi(freq)
    np.testing.assert_allclose(midi, midi_expected)
