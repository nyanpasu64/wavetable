from typing import List

import numpy as np

from wavetable import fourier
from wavetable.instrument import Instr
from wavetable.util.reprmixin import ReprMixin


class TransferFunctor(ReprMixin):
    # def __str__(self):
    #     return '%s(...)' % (type(self).__name__)

    def __call__(self, omega):
        raise NotImplementedError

    def __mul__(self, other):
        a, b = self, other

        class ProductFunctor(TransferFunctor):
            def __str__(self):
                return '(%s * %s)' % (a, b)

            def __call__(self, omega):
                return a(omega) * b(omega)

        return ProductFunctor()


def T(func):
    class TransferWrapper(TransferFunctor):
        def __str__(self):
            return 'T(%s)' % (func)

        __repr__ = __str__

        def __call__(self, omega):
            return func(omega)
    return TransferWrapper()


class LowF(TransferFunctor):
    def __init__(self, omega0, y):
        self.omega0 = omega0
        self.y = y

    def __call__(self, omega):
        if omega < self.omega0:
            return self.y * 1.0
        # elif omega == self.omega0:
        #     return (1 + self.y) / 2.0
        else:
            return 1.0


class HighF(TransferFunctor):
    def __init__(self, omega0, y):
        self.omega0 = omega0
        self.y = y

    def __call__(self, omega):
        if omega > self.omega0:
            return self.y * 1.0
        # elif omega == self.omega0:
        #     return (1 + self.y) / 2.0
        else:
            return 1.0


class BandF(TransferFunctor):
    def __init__(self, omegaL, omegaR, y):
        self.omegaL = omegaL
        self.omegaR = omegaR
        self.y = y

    def __call__(self, omega):
        if self.omegaL < omega < self.omegaR:
            return self.y * 1.0
        # elif omega in [self.omegaL, self.omegaR]:
        #     return (1 + self.y) / 2.0
        else:
            return 1.0


class Unity(TransferFunctor):
    def __call__(self, omega):
        return 1


def BandF2(omegaL, omegaR, y):
    return HighF(omegaL, y) * HighF(omegaR, 1 / y)


class LowPass1(TransferFunctor):
    def __init__(self, f_c, phase_delay=False):
        """ cutoff frequency... Should I add phase delay?"""
        self.f_c = f_c
        self.phase_delay = phase_delay

    def __call__(self, f):
        transfer = 1 / (1 + 1j*f/self.f_c)
        if self.phase_delay:
            return transfer
        else:
            return np.abs(transfer)


# **** Utility functions ****

WaveType = 'np.ndarray'
SpectrumType = 'np.ndarray'


def filter_wave(wave: WaveType, transfer) -> WaveType:
    harm = fourier.rfft_norm(wave)
    harm[1:] *= transfer(np.arange(1, len(harm)))
    return fourier.irfft_norm(harm)


def filter_waves(seq: List[WaveType], transfer) -> List[WaveType]:
    new_waves = []
    for wave in seq:
        new_wave = filter_wave(wave, transfer)
        new_waves.append(new_wave)

    return new_waves


def filter_instr(instr: 'Instr', transfer):
    """ Filters instr.waves with "transfer", in-place. """
    # """ Returns shallow copy of instr, with altered waves. """

    new_waves = filter_waves(instr.waves, transfer)
    instr.waves = new_waves

    # new_instr = copy(instr)
    # new_instr.waves = new_waves
    # return new_instr
