import math
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from typing import Tuple, Sequence, Optional, Union, List, Callable

import click
from dataclasses import dataclass, InitVar
from ruamel.yaml import YAML
from waveform_analysis.freq_estimation import freq_from_autocorr

from wavetable.dsp import fourier, wave_util, transfers
from wavetable.dsp.fourier import rfft_length, zero_pad, SpectrumType
from wavetable.dsp.wave_util import Rescaler
from wavetable.inputs.wave import load_wave
from wavetable.instrument import Instr, LOOP, RELEASE
from wavetable.merge import Merge
from wavetable.util.config import Alias, ConfigMixin
from wavetable.util.math import nearest_sub_harmonic, midi2ratio, midi2freq
from wavetable.util.parsing import safe_eval

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

assert transfers  # module used by cfg.transfer
# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
# np.loge = np.ln = np.log


Folder = click.Path(exists=True, file_okay=False)
CFG_EXT = '.n163'
yaml = YAML()


@click.command()
@click.argument('WAV_DIRS', type=Folder, nargs=-1, required=True)
@click.argument('DEST_DIR', type=Folder)
def main(wav_dirs: Sequence[str], dest_dir: str):
    """
    config.n163 is a YAML file:

    file: "filename.wav"    # Optionally rename to "filename.pitch.wav" and omit root_pitch
                            # (eg: piano.60.wav)
    root_pitch: 83          # MIDI pitch, middle C4 is 60, C5 is 72.
    [recommended] sweep: "0:15 | 15:30 30:15"
        # The program generates synchronized wave and volume envelopes. DO NOT EXCEED 0:64 OR 63:0.
        # 0 1 2 ... 13 14 | 15 16 ... 29 30 29 ... 17 16
    nsamp: 64

    [optional] nwave: 33    # Truncates output to first `nwave` frames. DO NOT EXCEED 64.
    [optional] fps: 60      # default=60
    [optional] fft_mode: zoh
        # "zoh" adds a high-frequency boost to compensate for N163 hardware.
        # "normal" does not.
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

    file_cfg = recursive_load_yaml(cfg_path)
    cfg = n163_cfg(**file_cfg)

    # dest
    dest_path = dest_dir / (cfg_path.name + '.txt')
    with dest_path.open('w') as f:
        with redirect_stdout(f):
            read = WaveReader(cfg_dir, cfg)
            instr = read.read()

            note = cfg.root_pitch
            instr.print(note)


def recursive_load_yaml(cfg_path, parents=None):
    if parents is None:
        parents = []

    cfg_path = Path(cfg_path).resolve()
    if cfg_path in parents:
        raise ValueError(f'infinite recursion detected: {parents} -> {cfg_path}')
    parents.append(cfg_path)

    file_cfg: dict = yaml.load(cfg_path)

    # Inheritance
    if 'include' in file_cfg:
        include = file_cfg['include']
        del file_cfg['include']

        include_cfg = recursive_load_yaml(include, parents)
        for k, v in include_cfg.items():
            file_cfg.setdefault(k, v)

    return file_cfg


def unrounded_cfg(**kwargs):
    kwargs.setdefault('range', None)
    kwargs.setdefault('vol_range', None)
    return WaveReaderConfig.new(kwargs)


def n163_cfg(**kwargs):
    return WaveReaderConfig.new(kwargs)


@dataclass
class WaveReaderConfig(ConfigMixin):
    nsamp: int
    root_pitch: float = None
    pitch_estimate = Alias('root_pitch')
    strict_pitch: bool = False

    wav_path: InitVar[str] = None
    path = Alias('wav_path')
    file = Alias('wav_path')
    files: List[Union[dict, 'FileConfig']] = None

    # Frame rate and subsampling
    fps: float = 60
    wave_sub: int = 1   # Each wave is repeated `wave_sub` times.
    env_sub: int = 1    # Each volume/frequency entry is repeated `env_sub` times.

    # Instrument subsampling via user-chosen indexes
    nwave: Optional[int] = None
    start: int = 0          # Deprecated
    before_end: int = 0

    sweep: Union[str, list] = ''        # FTI playback indices. Unused waves will be removed.

    # STFT configuration
    mode: InitVar[str] = 'stft'  # File.get_ffts_freqs
    fft_mode: str = 'zoh'
    stft_merge: str = 'power'
    width_ms: float = '1000 / 30'  # Length of each STFT window
    width_s: float = None
    transfer: str = 'transfers.Unity()'
    phase_f: Optional[str] = None
    phase: Union[str, float] = None

    # Output bit depth and rounding
    range: Optional[int] = 16
    vol_range: Optional[int] = 16

    def __post_init__(self, wav_path, mode):
        if wav_path is not None:
            self.root_pitch = parse_pitch(self.root_pitch, wav_path, 'root_pitch')
            self.files = [FileConfig(wav_path, self.root_pitch, mode=mode)]
        else:
            self.files = [FileConfig.new(file_info) for file_info in self.files]
            if self.root_pitch is None:
                if len(self.files) == 1:
                    self.root_pitch = self.files[0].pitch_estimate
                else:
                    raise TypeError(
                        'must specify root_pitch when providing multiple files')

        if self.width_s is not None:
            self.width_s = safe_eval(self.width_s)
            self.width_ms = 1000 * self.width_s
        else:
            self.width_ms = safe_eval(self.width_ms)
        self.fps = safe_eval(self.fps)
        self.sweep = parse_sweep(self.sweep)


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


def parse_pitch(pitch: Optional[float], wav_path: str, why: str) -> float:
    """ Obtain MIDI pitch from filename.
    Examples:
    - strings.69.wav -> 69
    - strings.69.5.wav -> 69.5
    """
    if pitch is not None:
        return safe_eval(pitch)

    wav_no_ext = Path(wav_path).stem
    if '.' not in wav_no_ext:
        raise TypeError(f"{why} parameter missing, not specified in wave name "
                        f"(try '{wav_no_ext}.60.wav')")

    # Trim until first period
    wav_meta = wav_no_ext.split('.', 1)[1]

    pitch = safe_eval(wav_meta)
    return pitch


@dataclass
class FileConfig(ConfigMixin):
    """ A single WAV file. Each WaveReaderConfig can hold multiple FileConfigs. """
    path: str
    pitch_estimate: float = None

    channel: int = None
    volume: float = 1.0
    speed: int = 1
    repitch: int = 1
    mode: str = 'stft'  # [stft, cycle]

    def __post_init__(self):
        self.pitch_estimate = parse_pitch(
            self.pitch_estimate, self.path, 'files[].pitch_estimate')
        self.volume = safe_eval(self.volume)


class File:
    segment_smp: int = None
    segment_time: float = None
    window: np.ndarray = None

    # The approximate ratio of output/input amplitude (in the absence of compensation).
    # See "tests/stft_volume_scaling/analyze_stft_volume.py" for details.
    VOLUME_RATIO = 0.585

    def __init__(self,
                 cfg_dir: Path,
                 cfg: FileConfig,
                 wcfg: WaveReaderConfig,
                 power_sum: Callable[[Sequence[complex]], complex]):
        """ Wraps a single .wav file, and extracts periodic FFTs at specified times. """

        self.power_sum = power_sum
        self.cfg = cfg
        self.wcfg = wcfg

        wav_path = str(cfg_dir / cfg.path)
        self.smp_s, self.wav = load_wave(wav_path)

        # Scale by volume
        self.wav = self.wav.astype(float) * cfg.volume
        if cfg.channel is not None:
            self.wav = self.wav[:, [cfg.channel]]

        self.fundamental_freq = midi2freq(cfg.pitch_estimate)

        # Shift speed to multiple of root pitch.
        speed_shift = 1 / midi2ratio(cfg.pitch_estimate - wcfg.root_pitch) * cfg.speed
        self.smp_s *= speed_shift

        # Multiply fundamental frequency to ensure FFT doesn't skip bins.
        self.fundamental_freq *= speed_shift

        # Zero-space the periodic FFT to create a harmonic.
        self.freq_mul = cfg.speed * cfg.repitch

        # Compute segment size and window
        segment_smp = self.smp_time(wcfg.width_ms / 1000)
        segment_smp = 2 ** math.ceil(np.log2(segment_smp))  # type: int
        if segment_smp == 0:
            raise ValueError('invalid width_ms/s, segment_smp == 0')
        self.segment_smp = segment_smp

        self.segment_time = self.time_smp(segment_smp)

        if self.cfg.mode == 'stft':
            # For BRR encoding, ensure the output amplitude is approximately unity-gain,
            # by dividing by estimated attenuation factor.
            self.window = np.hanning(segment_smp) / self.VOLUME_RATIO

    def get_ffts_freqs(self, time: float):
        """ Returns one (periodic FFT, frequency Hz)) per channel. """
        # Loop over channels
        return [self._get_periodic_fft_freq(data) for data in self._channel_data_at(time)]

    def _channel_data_at(self, time: float):
        sample_offset = self.smp_time(time)

        if sample_offset + self.segment_smp >= len(self.wav):
            sample_offset = len(self.wav) - self.segment_smp
        data_channel = self.wav[sample_offset : sample_offset + self.segment_smp]  # type: np.ndarray
        return data_channel.T.copy()

    def _get_periodic_fft_freq(self, data) -> Tuple[SpectrumType, float]:
        """ Returns periodic FFT and frequency (Hz) of data. """

        # Number of samples
        nsamp = self.wcfg.nsamp
        freq_mul = self.freq_mul

        mode = self.cfg.mode

        periodic_fft = []
        fundamental_bin: float = self._get_fundamental_bin(data)

        if mode == 'stft':
            # Get STFT.
            stft = self._stft(data)

            # Convert STFT to periodic FFT.
            for harmonic in range(rfft_length(nsamp, freq_mul)):
                begin = fundamental_bin * (harmonic - 0.5)
                end = fundamental_bin * (harmonic + 0.5)

                bands = stft[math.ceil(begin):math.ceil(end)]
                amplitude: complex = self.power_sum(bands)
                periodic_fft.append(amplitude)

        elif mode == 'cycle':
            period = round(len(data) / fundamental_bin)

            # Pick 1 period of data, from the middle of the region.
            end = (len(data) + period) // 2
            begin = end - period

            periodic_fft = fourier.rfft_norm(data[begin:end])

        else:
            raise ValueError(f'mode=[stft, cycle] (you supplied {mode})')

        # Multiply pitch of FFT.
        freq_mul_fft = zero_pad(periodic_fft, freq_mul)

        # Ensure we didn't omit any harmonics <= Nyquist.
        if mode == 'stft':
            fft_plus_harmonic_length = len(freq_mul_fft) + freq_mul
            assert fft_plus_harmonic_length > rfft_length(nsamp), \
                f'fft len={len(freq_mul_fft)} + {freq_mul} not > {rfft_length(nsamp)}'

        # cyc/s = cyc/bin * bin/samp * samp/s
        freq_hz = fundamental_bin / len(data) * self.smp_s
        return freq_mul_fft, freq_hz

    def _stft(self, data: np.ndarray) -> np.ndarray:
        """ Phasor phases will match center of data, or peak of window. """
        data *= self.window
        phased_data = np.roll(data, len(data) // 2)
        return fourier.rfft_norm(phased_data)

    def _get_fundamental_bin(self, data) -> float:
        """ Return estimated frequency of data. Units = STFT bins."""
        # cyc/window = cyc/s * s/window
        approx_bin = self.fundamental_freq * self.segment_time
        if self.wcfg.strict_pitch:
            return float(approx_bin)

        try:
            fft_peak = freq_from_autocorr(data, len(data))
        except IndexError:  # unhandled exception from library code
            fft_peak = 0

        freq_bin = nearest_sub_harmonic(fft_peak, approx_bin)

        return float(freq_bin)

    def smp_time(self, time):
        return int(time * self.smp_s)

    def time_smp(self, sample):
        return sample / self.smp_s


class WaveReader:
    ntick: Optional[int]

    def __init__(self, cfg_dir: Path, cfg: WaveReaderConfig):
        assert cfg_dir.is_dir()
        self.cfg = cfg

        self.freq_estimate = midi2freq(cfg.root_pitch)
        self.ntick = None

        # STFT parameters
        stft_merge = cfg.stft_merge
        if stft_merge == 'power':
            self.power_sum = wave_util.power_merge
        elif stft_merge == 'sum':
            self.power_sum = np.sum
        else:
            raise ValueError(f'stft_merge=[power, sum] (you supplied {stft_merge})')

        # Load wav files
        self.files = []
        for file_cfg in cfg.files:
            file = File(cfg_dir, file_cfg, cfg, self.power_sum)
            if file.wav.shape[1] > 1 and not self.cfg.range:
                print("Warning: Multi-channel wave will be amplified.")
                print("To prevent this, assign a wave amplitude 'range' or add 'channel=0'.")
            self.files.append(file)

        self.transfer = eval(cfg.transfer)

        # TODO switch from dataclasses to attrs, implement these as converters
        if cfg.phase_f:
            self.phase_f = eval(cfg.phase_f)

        # converter should use functools.singledispatch
        elif cfg.phase is not None:
            # We want to handle both numbers and expressions.
            if isinstance(cfg.phase, str):
                phase = cfg.phase
            else:
                phase = repr(cfg.phase)

            self.phase_f = lambda f: eval(phase, {
                **globals(),
                'f': f,
                'pi': np.pi,
                'saw': -np.pi / 2
            })

        else:
            self.phase_f = None

        # STFT mode
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

        # Calculate start_samp, stop_samp
        start_frame = self.cfg.start

        if self.cfg.nwave:
            stop_frame = start_frame + self.cfg.nwave
        else:
            durations = [len(file.wav) / file.smp_s for file in self.files]
            stop_secs = max(durations)
            stop_frame = self.frame_time(stop_secs) - self.cfg.before_end

        # read_at() for every frame in the audio file.
        frames = range(start_frame, stop_frame)
        self.ntick = len(frames)
        instr = self.read_at(frames)

        # Pick a subset of the waves extracted.
        if self.cfg.sweep:
            instr = instr[self.cfg.sweep]

        # Apply subsampling.
        instr.sweep = instr.sweep[::self.cfg.wave_sub]

        instr.vols = instr.vols[::self.cfg.env_sub]
        instr.freqs = instr.freqs[::self.cfg.env_sub]

        instr.remove_unused_waves()
        return instr

    def read_at(self, frames: Sequence) -> Instr:
        """ Read and align waves at specified samples, returning an Instr.
        read() calls read_at() for every frame in the audio file. """
        waves = []
        freqs = []
        vols = []
        for frame in frames:
            wave, freq, peak = self._wave_at(frame)
            waves.append(wave)
            freqs.append(freq)
            vols.append(peak)
        del peak

        waves = wave_util.align_waves(waves)
        global_peak = None
        if self.cfg.vol_range:
            vols, global_peak = self.vol_rescaler.rescale_peak(vols)
        return Instr(waves, freqs=freqs, vols=vols, peak=global_peak)

    def _wave_at(self, frame: int) -> Tuple[np.ndarray, float, float]:
        """ Pure function, no side effects.

        loop channels: computes frequency.
        loop channels: computes STFT and periodic FFT.

        Merges FFTs and creates wave. Rounds wave to cfg.range.

        Returns wave, frequency, and volume.
        """

        time = self.time_frame(frame)

        # Compute periodic FFTs.
        ffts = []
        freqs = []  # Estimated frequency, in Hz.
        for file in self.files:
            for fft, freq in file.get_ffts_freqs(time):
                ffts.append(fft)
                # fundamental frequency, not multiplied frequency
                freqs.append(freq / file.freq_mul)
                del fft, freq

        # Find average FFT.
        avg_fft = self.merger.merge_ffts(ffts, self.transfer)
        avg_freq_hz = np.mean(freqs)  # type: float

        # Reassign phases.
        if self.phase_f:
            # Don't rephase the DC bin.
            first_bin = 1

            phases = self.phase_f(np.arange(first_bin, len(avg_fft)))
            phasors = np.exp(1j * phases)

            avg_fft = np.abs(avg_fft).astype(complex)
            avg_fft[first_bin:] *= phasors

        # Create periodic wave.
        wave = self.irfft(avg_fft, self.cfg.nsamp)
        if self.cfg.range:
            wave, peak = self.rescaler.rescale_peak(wave)
        else:
            peak = 1.0

        return wave, avg_freq_hz, peak

    def time_frame(self, frame):
        return frame / self.cfg.fps

    def frame_time(self, time):
        return int(time * self.cfg.fps)


if __name__ == '__main__':
    main()
