import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Tuple, Sequence, Optional, Union

import click
import numpy as np
from dataclasses import dataclass
from ruamel.yaml import YAML
from waveform_analysis.freq_estimation import freq_from_autocorr

from wavetable.dsp import fourier, gauss, wave_util, transfers
from wavetable.dsp.wave_util import Rescaler, midi2freq
from wavetable.inputs.wave import load_wave
from wavetable.merge import Merge
from wavetable.types.instrument import Instr, LOOP, RELEASE
from wavetable.util.math import nearest_sub_harmonic
from wavetable.util.parsing import safe_eval

assert transfers  # module used by cfg.transfer
# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
# np.loge = np.ln = np.log


Folder = click.Path(exists=True, file_okay=False)
CFG_EXT = '.n163'


@click.command()
@click.argument('WAV_DIRS', type=Folder, nargs=-1, required=True)
@click.argument('DEST_DIR', type=Folder)
def main(wav_dirs: Sequence[str], dest_dir: str):
    """
    config.n163 is a YAML file:

    file: "filename.wav"        # quotes are optional
    nsamp: 64
    pitch_estimate: 83          # MIDI pitch, middle C4 is 60, C5 is 72.
                                # This tool may estimate the wrong octave, if line is missing.
                                # Exclude if WAV file has pitch changes 1 octave or greater.
    at: "0:15 | 15:30 30:15"    # The program generates synchronized wave and volume envelopes. DO NOT EXCEED 0:64 OR 63:0.
                                # 0 1 2 ... 13 14 | 15 16 ... 29 30 29 ... 17 16
                                # TODO: 0:30:10 should produce {0 0 0 1 1 1 ... 9 9 9} (30 items), mimicing FamiTracker behavior.
    [optional] nwave: 33        # Truncates output to first `nwave` frames. DO NOT EXCEED 64.
    [optional] fps: 240         # Increasing this value will effectively slow the wave down, or transpose the WAV downards. Defaults to 60.
    [optional] fft_mode: normal # "zoh" adds a high-frequency boost to compensate for N163 hardware, which may or may not increase high-pitched aliasing sizzle.
    """
    dest_dir = Path(dest_dir)

    for wav_dir in wav_dirs:
        wav_dir = Path(wav_dir)
        print(wav_dir)

        cfgs = sorted(cfg_path for cfg_path in wav_dir.iterdir()
                      if cfg_path.suffix == CFG_EXT and cfg_path.is_file())

        if not cfgs:
            raise click.ClickException(f'Wave directory {wav_dir} has no {CFG_EXT} files')

        for cfg_path in cfgs:
            print(cfg_path)
            process_cfg(cfg_path, dest_dir)


def process_cfg(cfg_path: Path, dest_dir: Path):
    # cfg
    cfg_path = cfg_path.resolve()
    cfg_dir = cfg_path.parent

    yaml = YAML(typ='safe')
    file_cfg = yaml.load(cfg_path)
    cfg = WaveReaderConfig(**file_cfg)

    # dest
    dest_path = dest_dir / (cfg_path.name + '.txt')
    with dest_path.open('w') as f:
        with redirect_stdout(f):
            read = WaveReader(cfg_dir, cfg)
            instr = read.read()

            note = cfg.pitch_estimate
            instr.print(note)


@dataclass
class WaveReaderConfig:
    wav_path: str
    pitch_estimate: int     # TODO documentation allows for excluding, but instr.print(note) fails?
    nsamp: int

    # Frame rate and subsampling
    fps: float = 60
    wave_sub: int = 1   # Each wave is repeated `wave_sub` times.
    env_sub: int = 1    # Each volume/frequency entry is repeated `env_sub` times.
    # subsampling: int = field(init=False)  TODO

    # Instrument subsampling via user-chosen indexes
    nwave: Optional[int] = None
    start: int = 0          # Deprecated
    before_end: int = 0

    # wave_locs: Union[str, list] = ''    # WAV recording indices. `sweep` will be autogenerated.
    sweep: Union[str, list] = ''        # FTI playback indices. Unused waves will be removed.

    # STFT configuration
    fft_mode: str = 'normal'
    width_ms: float = '1000 / 30'  # Length of each STFT window
    transfer: str = 'transfers.Unity()'
    phase_f: Optional[str] = None

    # Output bit depth and rounding
    range: Optional[int] = 16
    vol_range: Optional[int] = 16

    def __post_init__(self):
        self.width_ms = safe_eval(str(self.width_ms))
        self.sweep = parse_sweep(self.sweep)
        # self.subsampling = math.gcd(self.wave_sub, self.env_sub)  TODO


def parse_sweep(at: str) -> list:
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
    return WaveReaderConfig(**kwargs)


def n163_cfg(**kwargs):
    return WaveReaderConfig(**kwargs)


class WaveReader:
    ntick: Optional[int]

    def __init__(self, cfg_dir: Path, cfg: WaveReaderConfig):
        self.cfg = cfg

        assert cfg_dir.is_dir()
        self.cfg_dir = cfg_dir

        wav_path = str(cfg_dir / cfg.wav_path)

        # Load WAV file
        self.sr, self.wav = load_wave(wav_path)

        if cfg.pitch_estimate:
            self.freq_estimate = midi2freq(cfg.pitch_estimate)
        else:
            self.freq_estimate = None

        self.frame_time = 1 / cfg.fps
        self.ntick = None

        # STFT parameters
        segment_time = cfg.width_ms / 1000
        self.segment_smp = self.smp_time(segment_time)
        self.segment_smp = 2 ** math.ceil(np.log2(self.segment_smp))  # type: int
        self.segment_time = self.time_smp(self.segment_smp)

        self.window = np.hanning(self.segment_smp)
        self.power_sum = wave_util.power_merge
        self.transfer = eval(cfg.transfer)
        if cfg.phase_f:
            self.phase_f = eval(cfg.phase_f)
        else:
            self.phase_f = None

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

        # Channel merger (arguments irrelevant since we only use merge_ffts())
        self.merger = Merge(maxrange=None, fft='v1')

    def read(self) -> Instr:
        """ read_at() one wave per frame.
        Filter the results using `sweep`, `wave_sub`, and `env_sub`.
        Remove unused waves. """

        nsamp_frame = np.rint(self.smp_time(self.frame_time)).astype(int)

        # Calculate start_samp, stop_samp
        start_frame = self.cfg.start
        start_samp = start_frame * nsamp_frame

        if self.cfg.nwave:
            stop_samp = (start_frame + self.cfg.nwave) * nsamp_frame
        else:
            stop_samp = len(self.wav) - self.cfg.before_end * nsamp_frame

        # read_at() for every frame in the audio file.
        sample_offsets = list(range(start_samp, stop_samp, nsamp_frame))
        self.ntick = len(sample_offsets)
        instr = self.read_at(sample_offsets)

        # Pick a subset of the waves extracted.
        if self.cfg.sweep:
            instr = instr[self.cfg.sweep]

        # Apply subsampling.
        instr.sweep = instr.sweep[::self.cfg.wave_sub]

        instr.vols = instr.vols[::self.cfg.env_sub]
        instr.freqs = instr.freqs[::self.cfg.env_sub]

        instr.remove_unused_waves()
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
            vols, peak = self.vol_rescaler.rescale_peak(vols)
            print(f'peak = {peak}')
        return Instr(waves, freqs=freqs, vols=vols)

    def _wave_at(self, sample_offset: int) -> Tuple[np.ndarray, float, float]:
        """ Pure function, no side effects.

        loop channels: computes frequency.
        loop channels: computes STFT and periodic FFT.

        Merges FFTs and creates wave. Rounds wave to cfg.range.

        Returns wave, frequency, and volume.
        """

        data_channels = self.data_channels_at(sample_offset)
        channels_data = data_channels.T
        del data_channels

        stft_nsamp = len(channels_data[0])

        # Estimated frequency of each audio channel. Units = FFT bins.
        freq_bins = []
        for data in channels_data:
            if self.freq_estimate:
                # cyc/s * time/window = cyc/window
                approx_bin = self.freq_estimate * self.segment_time
                fft_peak = freq_from_autocorr(data, len(data))
                freq_bin = nearest_sub_harmonic(fft_peak, approx_bin)
            else:
                freq_bin = freq_from_autocorr(data, len(data))
            freq_bins.append(freq_bin)

        avg_freq_bin = np.mean(freq_bins)   # type: float

        # Calculate STFT of each channel.
        ffts = []
        for data in channels_data:
            freq_bin = avg_freq_bin     # TODO stereo uses average or channel freq?

            stft = self.stft(data)
            result_fft = []

            # Convert STFT to periodic FFT.
            for harmonic in range(gauss.nyquist_inclusive(self.cfg.nsamp)):
                begin = freq_bin * (harmonic - 0.5)
                end = freq_bin * (harmonic + 0.5)

                bands = stft[math.ceil(begin):math.ceil(end)]
                # if bands are uncorrelated, self.power_sum is better
                amplitude = np.sum(bands)   # type: complex
                result_fft.append(amplitude)

            ffts.append(result_fft)

        # Find average FFT.
        avg_fft = self.merger.merge_ffts(ffts, self.transfer)

        # Reassign phases.
        if self.phase_f:
            # Don't rephase the DC bin.
            first_bin = 1

            phases = self.phase_f(np.arange(first_bin, len(avg_fft)))
            phasors = np.exp(1j * phases)

            avg_fft = np.abs(avg_fft).astype(complex)
            avg_fft[first_bin:] *= phasors

        # Create periodic wave.
        wave = self.irfft(avg_fft)
        if self.cfg.range:
            wave, peak = self.rescaler.rescale_peak(wave)
        else:
            peak = 1

        freq_hz = avg_freq_bin / stft_nsamp * self.sr
        return wave, freq_hz, peak

    def stft(self, data: np.ndarray) -> np.ndarray:
        """ Phasor phases will match center of data, or peak of window. """
        data *= self.window
        phased_data = np.roll(data, len(data) // 2)
        return fourier.rfft_norm(phased_data)

    def data_channels_at(self, sample_offset):
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
    main()

#     # FIXME
#     message = 'Usage: %s config.n163' % Path(__file__).name + '''
#
# config.n163 is a YAML file:
#
# file: "filename.wav"        # quotes are optional
# nsamp: 64
# pitch_estimate: 83          # MIDI pitch, middle C4 is 60, C5 is 72.
#                             # This tool may estimate the wrong octave, if line is missing.
#                             # Exclude if WAV file has pitch changes 1 octave or greater.
# at: "0:15 | 15:30 30:15"    # The program generates synchronized wave and volume envelopes. DO NOT EXCEED 0:64 OR 63:0.
#                             # 0 1 2 ... 13 14 | 15 16 ... 29 30 29 ... 17 16
#                             # TODO: 0:30:10 should produce {0 0 0 1 1 1 ... 9 9 9} (30 items), mimicing FamiTracker behavior.
# [optional] nwave: 33        # Truncates output to first `nwave` frames. DO NOT EXCEED 64.
# [optional] fps: 240         # Increasing this value will effectively slow the wave down, or transpose the WAV downards. Defaults to 60.
# [optional] fft_mode: normal # "zoh" adds a high-frequency boost to compensate for N163 hardware, which may or may not increase high-pitched aliasing sizzle.
# '''
#     print(message, file=sys.stderr)
