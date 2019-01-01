from pathlib import Path
from typing import TYPE_CHECKING

from wavetable.to_brr import process_cfg, ExtractorCLI, \
    WAV_SUBDIR, yaml
from wavetable.util.fs import pushd

if TYPE_CHECKING:
    pass


def get_cfg_dict(nwave) -> dict:
    cfg_dict = yaml.load('''\
wav_path: Sample 19.wav
root_pitch: 74
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
    and writes brr files to global_cli.dest_dir.
    """

    with pushd('tests/test_waves'):
        global_cli = ExtractorCLI(dest_dir=Path('dest'))

        name = '19'
        cfg_dir = Path()
        cfg_path = (cfg_dir / name).with_suffix('.wtcfg')

        wav_dir = cfg_dir / WAV_SUBDIR
        brr_dir = global_cli.dest_dir

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

            process_cfg(global_cli, cfg_path)

            # Intermediate wav files in `cfg_dir/WAV_SUBDIR/cfg-012.wav`
            assert_count(wav_dir, '.wav')

            # brr files in `global_cli.dest_dir/cfg-012.brr`.
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
