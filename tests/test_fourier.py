# noinspection PyUnresolvedReferences
import numpy as np
from wavetable.dsp.fourier import _zero_pad, rfft_norm, irfft_norm, rfft_zoh, irfft_zoh


def assert_close(a, b):
    assert len(a) == len(b)
    assert np.allclose(a, b)


pulse = [1, 0, 0, 0]
pulse3 = pulse * 3
expected = [0.25] * 3

fft1 = rfft_norm(pulse)
assert_close(fft1, expected)


def test_irfft():
    zoh1 = rfft_norm(pulse)
    zoh3 = _zero_pad(zoh1, 3)
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
    zoh3 = _zero_pad(zoh1, 3)
    ipulse3 = irfft_zoh(zoh3)
    assert_close(ipulse3, pulse3)


long_pulse = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
FACTOR = 3
# pulse is 4, long_pulse is 4*3


def test_integration():
    zoh1 = rfft_zoh(long_pulse)
    assert np.allclose(zoh1.imag, 0)
    zoh1 = zoh1.real
    zoh3 = _zero_pad(zoh1, FACTOR)
    ipulse3 = irfft_zoh(zoh3, len(long_pulse))
    assert_close(ipulse3, pulse3)


def test_multiscale():
    short = rfft_zoh(pulse).real
    long = rfft_zoh(long_pulse).real
    assert_close(short, long[:len(short)])


""" One useful property is that fft([wave] * x) is identical to zero_pad(fft(wave), x). """
def test_repeat():
    cat = rfft_zoh(pulse * 3)
    pad = _zero_pad(rfft_zoh(pulse), 3)
    assert_close(cat, pad)
    assert not np.allclose(cat, rfft_norm(pulse * 3))
