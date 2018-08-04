import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Sequence, Optional

import click
import numpy as np
from ruamel.yaml import YAML
from scipy.io import wavfile

from wavetable.util.config import dataclass
from wavetable.wave_reader import WaveReader, WaveConfig


@dataclass
class ExtractorConfig:
    dest_dir: Path  # If using with wav2brr, it should end in '/~brr/'


yaml = YAML()

Folder = click.Path(exists=True, file_okay=False)
CFG_EXT = '.wtcfg'


@click.command()
@click.argument('WAV_DIRS', type=Folder, nargs=-1)
@click.argument('DEST_DIR', type=Folder)
def main(wav_dirs: Sequence[str], dest_dir: str):
    """Converts .wav files into a series of SNES-BRR wavetables compatible with AMK.

    WAV_DIRS: Contains .wtcfg and .wav files with the same name.
    DEST_DIR: Location where .brr files are written."""

    global_cfg = ExtractorConfig(dest_dir=Path(dest_dir))

    for wav_dir in wav_dirs:
        wav_dir = Path(wav_dir)

        cfgs = sorted(cfg_path for cfg_path in wav_dir.iterdir()
                      if cfg_path.suffix == CFG_EXT and cfg_path.is_file())

        if not cfgs:
            raise click.ClickException(f'Wave directory {wav_dir} has no .cfg files')

        for cfg_path in cfgs:
            process_cfg(global_cfg, cfg_path)


@dataclass
class WavetableConfig(WaveConfig):
    no_brr: bool = False
    unlooped_prefix: int = 0        # Controls the loop point of the wave.
    truncate_prefix: bool = True    # Remove unlooped prefix from non-initial samples
    gaussian: bool = True


DTYPE = np.int16
SAMPLE_RATE = 32000     # Doesn't actually matter for output BRR files.


def process_cfg(global_cfg: ExtractorConfig, cfg_path: Path):
    """
    Reads cfg_path. Writes intermediate wav files to a subdirectory of cfg_path,
    and writes brr files to global_cfg.dest_dir.

    The WAV file contains data + data[:unlooped_prefix].
    The BRR file loops WAV[unlooped_prefix:].
    """

    # Load file config
    file_cfg: dict = yaml.load(cfg_path)
    file_cfg.setdefault('range', None)
    file_cfg.setdefault('vol_range', None)

    cfg = WavetableConfig(**file_cfg)

    no_brr = cfg.no_brr
    unlooped_prefix = cfg.unlooped_prefix
    truncate_prefix = cfg.truncate_prefix
    gaussian = cfg.gaussian

    # Compute cfg-global parameters
    cfg_name = cfg_path.stem
    cfg_dir = cfg_path.parent
    brr_dir = global_cfg.dest_dir

    brr_cfg = BrrConfig(
        gaussian=gaussian,
        loop=unlooped_prefix,
    )

    # Load input
    wr = WaveReader(cfg_path.parent, cfg)
    instr = wr.read()

    # # TODO wav_dir brr_dir
    # # TODO clear all waves with name
    # # TODO clear all brrs with name
    #
    # wav_dir = get_wav_path(cfg_dir, '').parent      # type: Path
    # for wav in wav_dir.glob(f'{cfg_name}-*.wav'):
    #     wav.unlink()

    for i, wave in enumerate(instr.waveseq):
        wave_name = f'{cfg_name}-{i:03}'
        wav_path = get_wav_path(cfg_dir, wave_name)
        brr_path = get_brr_path(brr_dir, wave_name)

        wav_path.parent.mkdir(exist_ok=True)

        # Write WAV file
        wave = np.around(wave).astype(DTYPE)
        # Duplicate prefix of wave data
        wave = np.concatenate((wave, wave[:unlooped_prefix]))
        wavfile.write(str(wav_path), SAMPLE_RATE, wave)

        if no_brr:
            continue

        # Encode BRR file
        brr = BrrEncoder(brr_cfg, wav_path, brr_path)

        # The first sample's "prefix" is used. When we switch loop points to subsequent
        # samples, only their looped portions are used.
        if truncate_prefix and i != 0:
            brr.write(behead=True)
        else:
            brr.write()


SUBDIR = 'temp-wav'


def get_wav_path(prefix: Path, wave_name: str) -> Path:
    return prefix / SUBDIR / (wave_name + '.wav')


def get_brr_path(brr_dir: Path, wave_name: str) -> Path:
    return brr_dir / (wave_name + '.brr')


@dataclass
class BrrConfig:
    loop: Optional[int] = None
    truncate: Optional[int] = None

    ratio: float = 1.0
    volume: float = 1.0

    gaussian: bool = True
    nowrap: bool = True


class BrrEncoder:
    CMD = 'brr_encoder'
    LOOP_REGEX = re.compile(
        r'^Position of the loop within the BRR sample : \d+ samples = (\d+) BRR blocks\.',
        re.MULTILINE
    )
    RECIPROCAL_RATIO_REGEX = re.compile(
        r'Resampling by effective ratio of ([\d.]+)\.\.\.', re.MULTILINE)
    # Do not remove the trailing ellipses. That will hide bugs where the resampling
    # ratio is not extracted correctly (eg. truncated at the decimal point).

    def __init__(self, cfg: BrrConfig, wav_path: Path, brr_path: Path):
        self.cfg = cfg
        self.wav_path = wav_path
        self.brr_path = brr_path

    def write(self, behead=False) -> float:
        """ Writes a BRR file, returns actual resampling ratio """
        args = self._get_args()

        # Call brr_encoder
        result = subprocess.run(
            args, stdout=subprocess.PIPE,
            universal_newlines=True, encoding='latin-1'
        )
        output = result.stdout      # type: str

        # Parse output: loop points
        if self.cfg.loop is not None:
            loop_idx = int(search(self.LOOP_REGEX, output))
        else:
            loop_idx = 0
        byte_offset = loop_idx * 9

        # Write loop points into file.
        with self.brr_path.open('rb') as brr_file:
            data = brr_file.read()

            # Remove data before loop point.
            if behead:
                data = data[byte_offset:]
                byte_offset = 0

            # Write loop points and data.
            header_data = byte_offset.to_bytes(2, 'little') + data
            brr_file.truncate()
            brr_file.seek(0)
            brr_file.write(header_data)

        # Parse output: actual BRR resampling ratio
        wav2brr_ratio = 1 / Fraction(self.RECIPROCAL_RATIO_REGEX.search(output).group(1))
        return wav2brr_ratio


    def _get_args(self):
        cfg = self.cfg
        args = [self.CMD]

        if cfg.gaussian:
            args += ['-g']

        if cfg.nowrap:
            args += ['-w']

        # Loop and truncate
        if cfg.loop is not None:
            args[0:0] = ['-l' + str(cfg.loop)]
        if cfg.truncate is not None:
            args[0:0] = ['-t' + str(cfg.truncate)]

        # Resample
        # Even if ratio=1, encoder may resample slightly, to ensure looped is
        # multiple of 16. So enable bandlimited sinc to preserve high frequencies.
        # NOTE: Default linear interpolation is simple, but is garbage at
        # preserving high frequencies.
        args += [f'-rb{cfg.ratio}']

        # Attenuate volume
        if cfg.volume != 1.0:
            args += [f'-a{cfg.volume}']

        return args


def search(regex, s):
    return regex.search(s).group(1)


if __name__ == '__main__':
    main()
