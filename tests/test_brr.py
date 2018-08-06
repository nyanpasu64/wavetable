import os
from contextlib import contextmanager
from pathlib import Path
from typing import Union, TYPE_CHECKING

from wavetable.to_brr import process_cfg, ExtractorConfig, \
    WAV_SUBDIR, yaml

if TYPE_CHECKING:
    pass


@contextmanager
def pushd(new_dir: Union[Path, str]):
    previous_dir = os.getcwd()
    os.chdir(str(new_dir))
    try:
        yield
    finally:
        os.chdir(previous_dir)


def get_cfg_dict(nwave) -> dict:
    cfg_dict = yaml.load('''\
wav_path: Sample 19.wav
pitch_estimate: 74
nwave: ~

fps: 30
nsamp: 80
unlooped_prefix: 16
''')
    cfg_dict['nwave'] = nwave
    return cfg_dict


def test_wav_brr():
    """ process_cfg():
    Reads cfg_path. Writes intermediate wav files to cfg_path/WAV_SUBDIR,
    and writes brr files to global_cfg.dest_dir.
    """

    with pushd('tests/test_waves'):
        global_cfg = ExtractorConfig(dest_dir=Path('dest'))

        name = '19'
        cfg_dir = Path()
        cfg_path = (cfg_dir / name).with_suffix('.wtcfg')

        wav_dir = cfg_dir / WAV_SUBDIR
        brr_dir = global_cfg.dest_dir

        brr_dir.mkdir(exist_ok=True)
        del cfg_dir

        def assert_count(dir: Path, ext: str) -> None:
            """ Ensure there are exactly `nwave` files in the directory. """
            paths = dir.glob(f'{name}-*{ext}')
            paths = sorted(path.name for path in paths)
            assert paths == [f'{name}-{i:03}{ext}' for i in range(nwave)]

        # Ensure existing waves are deleted.
        for nwave in [2, 1]:
            yaml.dump(get_cfg_dict(nwave), cfg_path)

            process_cfg(global_cfg, cfg_path)

            # Intermediate wav files in `cfg_dir/WAV_SUBDIR/cfg-012.wav`
            assert_count(wav_dir, '.wav')

            # brr files in `global_cfg.dest_dir/cfg-012.brr`.
            assert_count(brr_dir, '.brr')


# TODO
# def test_brr_encoder():
#     brr_cfg = BrrConfig(
#         gaussian=gaussian,
#         loop=unlooped_prefix,
#     )
#
#     brr = BrrEncoder(cfg)
#     resampling_ratio = brr.write()
#     assert resampling_ratio == 1.0
