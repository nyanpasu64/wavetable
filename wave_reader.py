import math
from numbers import Number
from typing import Optional, List, Tuple, Sequence

import numpy as np
from scipy.io import wavfile
from waveform_analysis.freq_estimation import freq_from_autocorr
from wavetable import gauss
from wavetable.gauss import Rescaler
from wavetable.instrument import Instr
from wavetable.playback import pitch2freq

import fourier
from util import AttrDict

# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
np.loge = np.ln = np.log

DEFAULT_FPS = 60


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

        self.range = cfg.range  # type: Optional[int]
        if self.range:
            self.rescaler = Rescaler(self.range)

        self.nsamp = cfg.nsamp
        self.nwave = cfg.get('nwave', None)
        self.fps = cfg.get('fps', DEFAULT_FPS)
        self.frame_time = 1 / self.fps
        self.offset = cfg.get('offset', 0.5)

        self.mode = cfg.get('mode', 'stft')
        if self.mode == 'stft':
            # Maximum of 1/60th second or 2 periods
            segment_time = max(self.frame_time, 2 / self.freq_estimate)
            self.segment_smp = self.s_t(segment_time)
            self.segment_smp = 2 ** math.ceil(np.log2(self.segment_smp))    # type: int

            # self.stft = stfu(self.wav, fs=self.sr, nperseg=self.segment_smp, boundary=None)
            self.window = np.hanning(self.segment_smp)

    def s_t(self, time):
        return int(time * self.sr)

    def t_f(self, frame):
        return frame * self.frame_time

    def f_t(self, time):
        return int(time / self.frame_time)

    def t_s(self, sample):
        return sample / self.sr

    def raw_at(self, sample_offset):
        if sample_offset + self.segment_smp >= len(self.wav):
            sample_offset = len(self.wav) - self.segment_smp
        data = self.wav[sample_offset:sample_offset + self.segment_smp]     # type: np.ndarray
        return data.copy()

    def stft(self, sample_offset):
        """ Phasor phases will match center of data, or peak of window. """
        data = self.raw_at(sample_offset)
        data *= self.window
        phased_data = np.roll(data, len(data)//2)
        return np.fft.rfft(phased_data)

    def wave_at(self, sample_offset: int) -> Tuple[np.ndarray, float]:
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
                # FIXME calculate power
                result_fft.append(np.sum(bands))

            wave = fourier.irfft(result_fft)
            if self.range:
                wave = self.rescaler(wave)
            return wave, freq

    def read(self, start: int=1):
        """ For each frame, extract wave_at. """
        frame_dsamp = np.rint(self.s_t(self.frame_time)).astype(int)
        start_samp = start * frame_dsamp
        stop_samp = (start + self.nwave) * frame_dsamp  # len(self.wav)
        sample_offsets = list(range(start_samp, stop_samp, frame_dsamp))
        return self.read_at(sample_offsets)

    def read_at(self, sample_offsets: Sequence):
        waveseq = []
        freqs = []
        # TODO vols = []
        for offset in sample_offsets:
            wave, freq = self.wave_at(offset)
            waveseq.append(wave)
            freqs.append(freq)
        return Instr(waveseq, AttrDict(freqs=freqs))
