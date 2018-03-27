import math

import numpy as np
from scipy.io import wavfile
from waveform_analysis.freq_estimation import freq_from_autocorr
from wavetable import gauss
from wavetable.playback import pitch2freq

import fourier
from util import AttrDict

# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
np.loge = np.ln = np.log

_FPS = 60
FRAME_TIME = 1.0 / _FPS


def freq_from_fft_limited(signal, *, end):
    from waveform_analysis._common import parabolic
    from numpy.fft import rfft
    from numpy import argmax, log
    signal = np.asarray(signal)
    N = len(signal)

    # Compute Fourier transform of windowed signal
    windowed = signal * np.kaiser(N, 100)
    f = rfft(windowed)[:int(end)]
    f[0] = 1e-9

    # Find the peak and interpolate to get a more accurate peak
    i_peak = argmax(abs(f))  # Just use this value for less-accurate result
    # print(abs(f))
    # print(i_peak)
    i_interp = parabolic(log(abs(f)), i_peak)[0]
    # print(i_interp)

    return i_interp


class WaveReader:
    def __init__(self, path, cfg: AttrDict):
        self.path = path
        self.sr, self.wav = wavfile.read(path)    # type: int, np.ndarray
        self.wav = self.wav.astype(float)
        self.freq_estimate = pitch2freq(cfg.pitch_estimate)

        self.range = cfg.range
        self.nsamp = cfg.nsamp
        self.nwave = cfg.get('nwave', None)
        self.fps = cfg.get('fps', _FPS)
        self.offset = cfg.get('offset', 0.5)

        self.mode = cfg.get('mode', 'stft')
        if self.mode == 'stft':
            # Maximum of 1/60th second or 2 periods
            segment_time = max(FRAME_TIME, 2 / self.freq_estimate)
            self.segment_smp = self.s_t(segment_time)
            self.segment_smp = 2 ** math.ceil(np.log2(self.segment_smp))    # type: int

            # self.stft = stfu(self.wav, fs=self.sr, nperseg=self.segment_smp, boundary=None)
            self.window = np.hanning(self.segment_smp)

    def s_t(self, time):
        return int(time * self.sr)

    def t_f(self, frame):
        return frame * FRAME_TIME

    def f_t(self, time):
        return int(time / FRAME_TIME)

    def t_s(self, sample):
        return sample / self.sr

    def raw_at(self, sample_offset):
        data = self.wav[sample_offset:sample_offset + self.segment_smp]     # type: np.ndarray
        return data.copy()

    def stft(self, sample_offset):
        """ Phasor phases will match center of data, or peak of window. """
        data = self.raw_at(sample_offset)
        data *= self.window
        phased_data = np.roll(data, len(data)//2)
        return np.fft.rfft(phased_data)

    def wave_at(self, sample_offset):
        if self.mode == 'stft':
            # Get STFT. Extract ~~power~~ from bins into new waveform's Fourier buffer.

            data = self.raw_at(sample_offset)
            stft = self.stft(sample_offset)

            approx_freq = freq_from_autocorr(data, len(data)) # = self.freq_estimate
            freq = freq_from_fft_limited(data, end=1.5*approx_freq)

            result_fft = []

            for harmonic in range(gauss.nyquist_inclusive(self.nsamp)):
                # print(harmonic)
                begin = freq * (harmonic - 0.5)
                end = freq * (harmonic + 0.5)
                # print(begin, end)
                bands = stft[math.ceil(begin):math.ceil(end)]
                result_fft.append(np.sum(bands))

            return fourier.irfft(result_fft)


    def read(self):
        return Instr()
#         inds = range()
#         for sample_offset in np.arange(0, )
#         self.wave_at()
