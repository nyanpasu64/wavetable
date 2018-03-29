import math
import warnings
from contextlib import redirect_stdout
from typing import Optional, Tuple, Sequence
import sys

import numpy as np
from ruamel.yaml import YAML
from scipy.io import wavfile
from waveform_analysis.freq_estimation import freq_from_autocorr
from wavetable import fourier
from wavetable import gauss
from wavetable import wave_util
from wavetable.instrument import Instr
from wavetable.playback import pitch2freq
from wavetable.wave_util import AttrDict, Rescaler

# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
np.loge = np.ln = np.log

DEFAULT_FPS = 60


# TODO freq = fft / round(fft/autocorr)
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
    try:
        i_interp = parabolic(log(abs(f)), i_peak)[0]
        return i_interp
    except IndexError:
        return float(i_peak)


class WaveReader:
    def __init__(self, path, cfg: dict):
        cfg = AttrDict(cfg)

        self.path = path
        with warnings.catch_warnings():
            # Polyphone SF2 rips contain 'smpl' chunk with loop data
            warnings.simplefilter("ignore")
            self.sr, self.wav = wavfile.read(path)    # type: int, np.ndarray
            if self.wav.ndim > 1:
                self.wav = self.wav[:, 0]

        self.wav = self.wav.astype(float)
        self.freq_estimate = pitch2freq(cfg.pitch_estimate)

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

            self.window = np.hanning(self.segment_smp)
            self.power_sum = wave_util.power_merge
        else:
            raise NotImplementedError('only stft supported')

        self.range = cfg.range  # type: Optional[int]
        if self.range:
            self.rescaler = Rescaler(self.range)

        self.vol_range = cfg.get('vol_range', None)
        if self.vol_range:
            self.vol_rescaler = Rescaler(self.vol_range, translate=False)

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

    def wave_at(self, sample_offset: int) -> Tuple[np.ndarray, float, float]:
        """
        :param sample_offset: offset
        :return: (wave, freq, volume)
        """

        if self.mode == 'stft':
            # Get STFT. Extract ~~power~~ from bins into new waveform's Fourier buffer.

            data = self.raw_at(sample_offset)
            stft = self.stft(sample_offset)

            approx_freq = freq_from_autocorr(data, len(data)) # = self.freq_estimate
            freq_bin = freq_from_fft_limited(data, end=1.5*approx_freq)

            result_fft = []

            for harmonic in range(gauss.nyquist_inclusive(self.nsamp)):
                # print(harmonic)
                begin = freq_bin * (harmonic - 0.5)
                end = freq_bin * (harmonic + 0.5)
                # print(begin, end)
                bands = stft[math.ceil(begin):math.ceil(end)]
                amplitude = self.power_sum(bands)
                result_fft.append(amplitude)

            wave = fourier.irfft(result_fft)
            if self.range:
                wave, peak = self.rescaler.rescale_peak(wave)
            else:
                peak = 1

            freq_hz = freq_bin / len(data) * self.sr

            return wave, freq_hz, peak

    def read(self, start: int=1):
        """ For each frame, extract wave_at. """
        frame_dsamp = np.rint(self.s_t(self.frame_time)).astype(int)
        start_samp = start * frame_dsamp
        if self.nwave:
            stop_samp = (start + self.nwave) * frame_dsamp
        else:
            stop_samp = len(self.wav)

        sample_offsets = list(range(start_samp, stop_samp, frame_dsamp))
        return self.read_at(sample_offsets)

    def read_at(self, sample_offsets: Sequence):
        wave_seq = []
        freqs = []
        vols = []
        for offset in sample_offsets:
            wave, freq, peak = self.wave_at(offset)
            wave_seq.append(wave)
            freqs.append(freq)
            vols.append(peak)

        wave_seq = wave_util.align_waves(wave_seq)
        if self.vol_range:
            vols = self.vol_rescaler.rescale(vols)
        return Instr(wave_seq, AttrDict(freqs=freqs, vols=vols))


def n163_cfg():
    return AttrDict(
        range=16,
        vol_range=16,
        fps=60
    )


def unrounded_cfg():
    return AttrDict(
        range=None,
        vol_range=None,
        fps=60
    )


def main():
    default = n163_cfg()

    cfg_path = sys.argv[1]

    yaml = YAML(typ='safe')
    with open(cfg_path) as f:
        cfg = yaml.load(f)
    print(cfg)

    default.update(cfg)
    cfg = default

    path = cfg['file']
    with open(path + '.txt', 'w') as f:
        with redirect_stdout(f):
            read = WaveReader(path, cfg)
            instr = read.read()

            note = cfg['pitch_estimate']
            instr.print(note)


if __name__ == '__main__':
    main()
