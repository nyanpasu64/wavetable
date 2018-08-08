import math
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Tuple, Sequence, Optional

import numpy as np
from dataclasses import dataclass, field, InitVar
from ruamel.yaml import YAML
from scipy.io import wavfile
from waveform_analysis.freq_estimation import freq_from_autocorr

from wavetable import fourier, transfers
from wavetable import gauss
from wavetable import wave_util
from wavetable.instrument import Instr
from wavetable.playback import midi2freq
from wavetable.util.parsing import safe_eval
from wavetable.wave_util import AttrDict, Rescaler

assert transfers    # module used by cfg.transfer
# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
# np.loge = np.ln = np.log


def main(cfg_path):
    cfg_path = Path(cfg_path).resolve()
    cfg_dir = cfg_path.parent

    yaml = YAML(typ='safe')
    file_cfg = yaml.load(cfg_path)

    cfg = WaveConfig(**file_cfg)

    with open(str(cfg_path) + '.txt', 'w') as f:
        with redirect_stdout(f):
            read = WaveReader(cfg_dir, cfg)
            instr = read.read()

            note = cfg.pitch_estimate
            instr.print(note)


@dataclass
class WaveConfig:
    wav_path: str
    pitch_estimate: int     # TODO documentation allows for excluding, but instr.print(note) fails?
    nsamp: int

    # Frame rate and subsampling
    fps: int = 60
    wave_sub: int = 1   # Each wave is repeated `wave_sub` times.
    env_sub: int = 1    # Each volume/frequency entry is repeated `env_sub` times.
    # subsampling: int = field(init=False)  TODO

    # Instrument subsampling via user-chosen indexes
    nwave: Optional[int] = None
    start: int = 0          # Deprecated

    at: InitVar[str] = ''   # Deprecated
    wave_indices: list = field(init=False)

    # STFT configuration
    fft_mode: str = 'normal'
    segment_ms: float = 1000/60  # Length of each STFT window
    transfer: str = 'transfers.Unity()'

    # Output bit depth and rounding
    range: Optional[int] = 16
    vol_range: Optional[int] = 16

    def __post_init__(self, at):
        self.segment_ms = safe_eval(str(self.segment_ms))
        self.wave_indices = parse_at(at or '')
        # self.subsampling = math.gcd(self.wave_sub, self.env_sub)  TODO


def parse_at(at: str):
    out = []
    for word in at.split():
        try:
            out.append(int(word, 0))
            continue
        except ValueError:
            pass

        if ':' in word:
            chunks = [int(pos, 0) if pos else None
                      for pos in word.split(':')]
            if len(chunks) == 2:
                chunks.append(None)

            try:
                if chunks[1] < chunks[0] and chunks[2] is None:
                    chunks[2] = -1
            except TypeError:
                pass

            out.append(slice(*chunks))
        elif word in [LOOP, RELEASE]:
            out.append(word)
        else:
            raise ValueError(f'Invalid sweep parameter {word}')
    return out


def unrounded_cfg(**kwargs):
    kwargs.setdefault('range', None)
    kwargs.setdefault('vol_range', None)
    return WaveConfig(**kwargs)


def n163_cfg(**kwargs):
    return WaveConfig(**kwargs)


class WaveReader:
    def __init__(self, cfg_dir: Path, cfg: WaveConfig):
        self.cfg = cfg

        assert cfg_dir.is_dir()
        self.cfg_dir = cfg_dir

        wav_path = str(cfg_dir / cfg.wav_path)

        # Load WAV file
        with warnings.catch_warnings():
            # Polyphone SF2 rips contain 'smpl' chunk with loop data
            warnings.simplefilter("ignore")
            self.sr, self.wav = wavfile.read(wav_path)  # type: int, np.ndarray
            if self.wav.ndim > 1:
                self.wav = self.wav[:, 0]   # TODO power_merge stereo samples

        self.wav = self.wav.astype(float)   # TODO divide by peak
        if cfg.pitch_estimate:
            self.freq_estimate = midi2freq(cfg.pitch_estimate)
        else:
            self.freq_estimate = None

        self.frame_time = 1 / cfg.fps
        # self.offset = cfg.get('offset', 0.5)

        # STFT parameters
        segment_time = cfg.segment_ms / 1000
        self.segment_smp = self.smp_time(segment_time)
        self.segment_smp = 2 ** math.ceil(np.log2(self.segment_smp))  # type: int
        self.segment_time = self.time_smp(self.segment_smp)

        self.window = np.hanning(self.segment_smp)
        self.power_sum = wave_util.power_merge
        self.transfer = eval(cfg.transfer)

        fft_mode = cfg.fft_mode
        if fft_mode == 'normal':
            self.irfft = fourier.irfft_norm
        elif fft_mode == 'zoh':
            self.irfft = fourier.irfft_zoh
        else:
            raise ValueError(f'fft_mode=[zoh, normal] (you supplied {fft_mode})')

        # Rescaling parameters
        if cfg.range:
            self.rescaler = Rescaler(cfg.range)

        if cfg.vol_range:
            self.vol_rescaler = Rescaler(cfg.vol_range, translate=False)

    def read(self) -> Instr:
        """ read_at() one wave per frame, then filter the results. """

        nsamp_frame = np.rint(self.smp_time(self.frame_time)).astype(int)

        # Calculate start_samp, stop_samp
        start_frame = self.cfg.start
        start_samp = start_frame * nsamp_frame

        if self.cfg.nwave:
            stop_samp = (start_frame + self.cfg.nwave) * nsamp_frame
        else:
            stop_samp = len(self.wav)

        # read_at() for every frame in the audio file.
        sample_offsets = list(range(start_samp, stop_samp, nsamp_frame))
        instr = self.read_at(sample_offsets)

        # Pick a subset of the waves extracted.
        if self.cfg.wave_indices:
            instr = instr[self.cfg.wave_indices]
        return instr

    def read_at(self, sample_offsets: Sequence) -> Instr:
        """ Read and align waves at specified samples, returning an Instr.
        read() calls read_at() for every frame in the audio file. """
        waves = []
        freqs = []
        vols = []
        for offset in sample_offsets:
            wave, freq, peak = self._wave_at(offset)
            waves.append(wave)
            freqs.append(freq)
            vols.append(peak)

        waves = wave_util.align_waves(waves)
        if self.cfg.vol_range:
            vols = self.vol_rescaler.rescale(vols)
        return Instr(waves, freqs=freqs, vols=vols)

    def _wave_at(self, sample_offset: int) -> Tuple[np.ndarray, float, float]:
        """ Pure function, no side effects.
        Computes STFT at sample, rounds wave to cfg.range.
        Returns wave, frequency, and volume.
        """

        # Get STFT. Extract ~~power~~ from bins into new waveform's Fourier buffer.

        data = self.raw_at(sample_offset)
        stft = self.stft(sample_offset)

        if self.freq_estimate:
            # cyc/s * time/window = cyc/window
            approx_bin = self.freq_estimate * self.segment_time
            fft_peak = freq_from_autocorr(data, len(data))
            harmonic = round(fft_peak / approx_bin)
            freq_bin = fft_peak / harmonic

        else:
            freq_bin = freq_from_autocorr(data, len(data))

        result_fft = []

        for harmonic in range(gauss.nyquist_inclusive(self.cfg.nsamp)):
            # print(harmonic)
            begin = freq_bin * (harmonic - 0.5)
            end = freq_bin * (harmonic + 0.5)
            # print(begin, end)
            bands = stft[math.ceil(begin):math.ceil(end)]
            # TODO if bands are uncorrelated, self.power_sum is better
            amplitude = np.sum(bands)   # type: complex
            if harmonic > 0:
                amplitude *= self.transfer(harmonic)
            result_fft.append(amplitude)

        wave = self.irfft(result_fft)
        if self.cfg.range:
            wave, peak = self.rescaler.rescale_peak(wave)
        else:
            peak = 1

        freq_hz = freq_bin / len(data) * self.sr

        return wave, freq_hz, peak

    def stft(self, sample_offset):
        """ Phasor phases will match center of data, or peak of window. """
        data = self.raw_at(sample_offset)
        data *= self.window
        phased_data = np.roll(data, len(data) // 2)
        return fourier.rfft_norm(phased_data)

    def raw_at(self, sample_offset):
        if sample_offset + self.segment_smp >= len(self.wav):
            sample_offset = len(self.wav) - self.segment_smp
        data = self.wav[sample_offset:sample_offset + self.segment_smp]  # type: np.ndarray
        return data.copy()

    def smp_time(self, time):
        return int(time * self.sr)

    def time_frame(self, frame):
        return frame * self.frame_time

    def frame_time(self, time):
        return int(time / self.frame_time)

    def time_smp(self, sample):
        return sample / self.sr


if __name__ == '__main__':
    if 1 < len(sys.argv):
        main(sys.argv[1])
    else:
        message = 'Usage: %s config.n163' % Path(__file__).name + '''

config.n163 is a YAML file:

file: "filename.wav"        # quotes are optional
nsamp: N163 wave length
pitch_estimate: 83          # MIDI pitch, middle C4 is 60, C5 is 72.
                                # This tool may estimate the wrong octave, if line is missing.
                                # Exclude if WAV file has pitch changes 1 octave or greater.
at: "0:15 | 15:30 30:15"    # The program generates synchronized wave and volume envelopes. DO NOT EXCEED 0:64 OR 63:0.
                                # 0 1 2 ... 13 14 | 15 16 ... 29 30 29 ... 17 16
                                # TODO: 0:30:10 should produce {0 0 0 1 1 1 ... 9 9 9} (30 items), mimicing FamiTracker behavior.
[optional] nwave: 33        # Truncates output to first `nwave` frames. DO NOT EXCEED 64.
[optional] fps: 240         # Increasing this value will effectively slow the wave down, or transpose the WAV downards. Defaults to 60.
[optional] fft_mode: normal # "zoh" adds a high-frequency boost to compensate for N163 hardware, which may or may not increase high-pitched aliasing sizzle.
'''
        print(message, file=sys.stderr)
