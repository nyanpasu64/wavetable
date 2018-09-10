import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Sequence, Optional, Pattern, AnyStr, List, Dict

import click
import numpy as np
from dataclasses import dataclass, asdict
from ruamel.yaml import YAML
from scipy.io import wavfile

from wavetable.wave_reader import WaveReader, WaveReaderConfig
from wavetable.util.math import freq2midi


@dataclass
class ExtractorConfig:
    dest_dir: Path  # If using with wav2brr, it should end in '/~brr/'


yaml = YAML()

Folder = click.Path(exists=True, file_okay=False)
CFG_EXT = '.wtcfg'
WAVETABLE_PATH = Path('wavetable.yaml')


@click.command()
@click.argument('WAV_DIRS', type=Folder, nargs=-1, required=True)
@click.argument('DEST_DIR', type=Folder)
def main(wav_dirs: Sequence[str], dest_dir: str):
    """Converts .wav files into a series of SNES-BRR wavetables compatible with AMK.

    WAV_DIRS: Contains .wtcfg and .wav files with the same name.
    DEST_DIR: Location where .brr files are written."""

    global_cfg = ExtractorConfig(dest_dir=Path(dest_dir))
    for wav_dir in wav_dirs:
        wav_dir = Path(wav_dir)
        print(wav_dir)

        cfgs = sorted(cfg_path for cfg_path in wav_dir.iterdir()
                      if cfg_path.suffix == CFG_EXT and cfg_path.is_file())

        if not cfgs:
            raise click.ClickException(f'Wave directory {wav_dir} has no {CFG_EXT} files')

        metadata_list: Dict[str, dict] = {}

        # wave_fps: 30
        # pitch_fps: 90
        # pitches: [60.1, 60.2...]

        # Process each .cfg file.
        for cfg_path in cfgs:
            print(cfg_path)
            cfg_name = cfg_path.stem
            metadata = process_cfg(global_cfg, cfg_path)
            metadata_list[cfg_name] = asdict(metadata)

        # Wavetable metadata file
        yaml.dump(metadata_list, WAVETABLE_PATH)


@dataclass
class WavetableConfig(WaveReaderConfig):
    # override default
    fft_mode: str = 'normal'

    no_brr: bool = False
    unlooped_prefix: int = 0        # Controls the loop point of the wave.
    truncate_prefix: bool = True    # Remove unlooped prefix from non-initial samples
    gaussian: bool = True


@dataclass
class WavetableMetadata:
    nsamp: int
    ntick: int
    fps: float
    wave_sub: int   # Each wave is repeated `wave_sub` times.
    env_sub: int    # Each volume/frequency entry is repeated `env_sub` times.

    pitches: List[float]


DTYPE = np.int16
SAMPLE_RATE = 32000     # Doesn't actually matter for output BRR files.
WAV_SUBDIR = 'temp-wav'


def process_cfg(global_cfg: ExtractorConfig, cfg_path: Path) -> WavetableMetadata:
    """
    Reads cfg_path.
    Writes intermediate wav files to `cfg_dir/WAV_SUBDIR/cfg-012.wav`.
    Writes brr files to `global_cfg.dest_dir/cfg-012.brr`.

    The WAV file contains data + data[:unlooped_prefix].
    The BRR file loops WAV[unlooped_prefix:].
    """
    if not cfg_path.is_file():
        raise ValueError(f'invalid cfg_path {cfg_path}, is not file')

    # Initialize directories
    cfg_name = cfg_path.stem

    cfg_dir = cfg_path.parent   # type: Path
    wav_dir = cfg_dir / WAV_SUBDIR
    brr_dir = global_cfg.dest_dir

    wav_dir.mkdir(exist_ok=True)

    for file in wav_dir.glob(f'{cfg_name}-*'):
        file.unlink()

    for file in brr_dir.glob(f'{cfg_name}-*'):
        file.unlink()

    # Load config file
    file_cfg: dict = yaml.load(cfg_path)
    file_cfg.setdefault('range', None)
    file_cfg.setdefault('vol_range', None)

    cfg = WavetableConfig(**file_cfg)

    no_brr = cfg.no_brr
    unlooped_prefix = cfg.unlooped_prefix
    truncate_prefix = cfg.truncate_prefix
    gaussian = cfg.gaussian

    brr_cfg = BrrEncoderConfig(
        gaussian=gaussian,
        loop=unlooped_prefix,
    )

    # Generate wavetables
    wr = WaveReader(cfg_path.parent, cfg)
    instr = wr.read()
    ntick = wr.ntick
    print(f'{cfg_name}: ntick={ntick}')

    for i, wave in enumerate(instr.waves):
        wave_name = f'{cfg_name}-{i:03}'

        wav_path = (wav_dir / wave_name).with_suffix('.wav')
        wav_path.parent.mkdir(exist_ok=True)

        # Write WAV file
        wave = np.around(wave).astype(DTYPE)
        # Duplicate prefix of wave data
        wave = np.concatenate((wave, wave[:unlooped_prefix]))
        wavfile.write(str(wav_path), SAMPLE_RATE, wave)

        if no_brr:
            continue

        # Encode BRR file
        brr_path = (brr_dir / wave_name).with_suffix('.brr')
        brr = BrrEncoder(brr_cfg, wav_path, brr_path)

        # The first sample's "prefix" is used. When we switch loop points to subsequent
        # samples, only their looped portions are used.
        if truncate_prefix and i != 0:
            brr.write(behead=True)
        else:
            brr.write()

    # Generate metadata

    pitches: List[float] = freq2midi(instr.freqs).tolist()

    return WavetableMetadata(
        nsamp=cfg.nsamp,
        ntick=ntick,
        fps=cfg.fps,
        wave_sub=cfg.wave_sub,
        env_sub=cfg.env_sub,
        pitches=pitches,
    )


@dataclass
class BrrEncoderConfig:
    loop: Optional[int] = None
    truncate: Optional[int] = None

    ratio: float = 1.0
    volume: float = 1.0

    gaussian: bool = True
    nowrap: bool = True
    filters: str = '01'
    # Predictors '01' have exponentially decaying error after swapping samples.
    # Higher-order predictors will cause the BRR hardware to extrapolate from
    # the wrong slope, leading to massive errors and/or overflow.


class BrrEncoder:
    _CMD = 'brr_encoder'
    _LOOP_REGEX = re.compile(
        r'^Position of the loop within the BRR sample : \d+ samples = (\d+) BRR blocks\.',
        re.MULTILINE
    )
    _RECIPROCAL_RATIO_REGEX = re.compile(
        r'Resampling by effective ratio of ([\d.]+)\.\.\.', re.MULTILINE)
    # Do not remove the trailing ellipses. That will hide bugs where the resampling
    # ratio is not extracted correctly (eg. truncated at the decimal point).

    def __init__(self, cfg: BrrEncoderConfig, wav_path: Path, brr_path: Path):
        self.cfg = cfg
        self.wav_path = wav_path
        self.brr_path = brr_path

        if not wav_path.is_file():
            raise ValueError(f'invalid wav_path {wav_path}, is not file')
        if brr_path.is_dir():
            raise ValueError(f'invalid brr_path {brr_path}, is dir')

    def write(self, behead=False) -> float:
        """ Writes a BRR file, returns actual resampling ratio """
        args = self._get_args()

        # Call brr_encoder
        result = subprocess.run(
            args, stdout=subprocess.PIPE,
            universal_newlines=True, encoding='latin-1'
        )
        result.check_returncode()
        output = result.stdout      # type: str

        # Parse output: loop points
        if self.cfg.loop is not None:
            loop_idx = int(search(self._LOOP_REGEX, output))
        else:
            loop_idx = 0
        byte_offset = loop_idx * 9

        # Write loop points into file.
        with self.brr_path.open('r+b') as brr_file:
            data = brr_file.read()

            # Remove data before loop point.
            if behead:
                data = data[byte_offset:]
                byte_offset = 0

            # Write loop points and data.
            header_data = byte_offset.to_bytes(2, 'little') + data
            brr_file.seek(0)
            brr_file.truncate()
            brr_file.write(header_data)

        # Parse output: actual BRR resampling ratio
        wav2brr_ratio = 1 / Fraction(search(self._RECIPROCAL_RATIO_REGEX, output))
        return wav2brr_ratio

    def _get_args(self):
        cfg = self.cfg
        args = [self._CMD]

        if cfg.gaussian:
            args += ['-g']

        if cfg.nowrap:
            args += ['-w']

        # Loop and truncate
        if cfg.loop is not None:
            args += ['-l' + str(cfg.loop)]
        if cfg.truncate is not None:
            args += ['-t' + str(cfg.truncate)]

        # Resample
        # Even if ratio=1, encoder may resample slightly, to ensure looped is
        # multiple of 16. So enable bandlimited sinc to preserve high frequencies.
        # NOTE: Default linear interpolation is simple, but is garbage at
        # preserving high frequencies.
        args += [f'-rb{cfg.ratio}']

        # Attenuate volume
        if cfg.volume != 1.0:
            args += [f'-a{cfg.volume}']

        # Filters
        if cfg.filters:
            args += [f'-f{cfg.filters}']

        # paths
        args += [str(self.wav_path), str(self.brr_path)]

        assert args[0] == self._CMD
        return args


def search(regex: Pattern, s: AnyStr) -> AnyStr:
    return regex.search(s).group(1)


if __name__ == '__main__':
    main()
