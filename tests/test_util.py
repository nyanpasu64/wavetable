import numpy as np

from wavetable.dsp.wave_util import A, freq2midi, midi2freq, midi2ratio


def test_freq2midi():
    a4 = 69
    octave = 12

    freq = A(440, 880, 220)
    midi_expected = A(a4, a4 + octave, a4 - octave)

    midi = freq2midi(freq)
    np.testing.assert_allclose(midi, midi_expected)


def test_midi2freq():
    assert midi2freq(69) == 440
    assert midi2freq(69-12) == 220


def test_midi2ratio():
    assert midi2ratio(12) == 2
    assert midi2ratio(-12) == .5
