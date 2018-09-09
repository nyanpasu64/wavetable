# noinspection PyUnresolvedReferences
import numpy as np
from wavetable.dsp.fourier import zero_pad, rfft_norm, irfft_norm, rfft_zoh, irfft_zoh, \
    nyquist_real_idx, rfft_length


def assert_close(a, b):
    assert len(a) == len(b)
    assert np.allclose(a, b)


pulse = [1, 0, 0, 0]
pulse3 = pulse * 3
expected = [0.25] * 3

fft1 = rfft_norm(pulse)
assert_close(fft1, expected)


def test_nyquist_real():
    assert nyquist_real_idx(4) == 2
    assert nyquist_real_idx(5) == 3


def test_nyquist_inclusive():
    assert rfft_length(4) == 3
    assert rfft_length(5) == 3

    # A 16-sample wave has harmonics [0..8] with length 9.
    assert rfft_length(16, 1) == 9

    # Bandlimiting its second harmonic produces harmonics [0,2,4,6,8] with length 5.
    assert rfft_length(16, 2) == 5

    # Bandlimiting its third harmonic produces harmonics [0,3,6] with length 3.
    assert rfft_length(16, 3) == 3



def test_irfft():
    zoh1 = rfft_norm(pulse)
    zoh3 = zero_pad(zoh1, 3)
    ipulse3 = irfft_norm(zoh3)
    assert_close(ipulse3, pulse3)


def test_rfft_zoh():
    zoh1 = rfft_zoh(pulse)
    assert len(zoh1) == len(expected)


def test_irfft_zoh():
    zoh1 = rfft_zoh(pulse)
    ipulse = irfft_zoh(zoh1)
    assert_close(ipulse, pulse)


def test_irfft_pad():
    zoh1 = rfft_zoh(pulse)
    zoh3 = zero_pad(zoh1, 3)
    ipulse3 = irfft_zoh(zoh3)
    assert_close(ipulse3, pulse3)


long_pulse = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
FACTOR = 3
# pulse is 4, long_pulse is 4*3


def test_integration():
    zoh1 = rfft_zoh(long_pulse)
    assert np.allclose(zoh1.imag, 0)
    zoh1 = zoh1.real
    zoh3 = zero_pad(zoh1, FACTOR)
    ipulse3 = irfft_zoh(zoh3, len(long_pulse))
    assert_close(ipulse3, pulse3)


def test_multiscale():
    short = rfft_zoh(pulse).real
    long = rfft_zoh(long_pulse).real
    assert_close(short, long[:len(short)])


""" One useful property is that fft([wave] * x) is identical to zero_pad(fft(wave), x). """
def test_repeat():
    cat = rfft_zoh(pulse * 3)
    pad = zero_pad(rfft_zoh(pulse), 3)
    assert_close(cat, pad)
    assert not np.allclose(cat, rfft_norm(pulse * 3))
