import io
from contextlib import redirect_stdout
from pathlib import Path

import dataclasses
import numpy as np
import pytest

from wavetable.types.instrument import Instr
from wavetable.util.math import ceildiv
from wavetable.util.fs import pushd
from wavetable.wave_reader import WaveReader, n163_cfg, unrounded_cfg, WaveReaderConfig, \
    recursive_load_yaml

CFG_DIR = Path('tests')
WAV_PATH = Path('test_waves/Sample 19.wav')
NSAMP = 128
NWAVE = 30


@pytest.fixture(scope="module", params=[n163_cfg, unrounded_cfg])
def cfg(request) -> WaveReaderConfig:
    """ request.param is a cfg factory, taken from params.
    "request" is a hardcoded name. """
    cfg = request.param(
        wav_path=WAV_PATH,
        nsamp=NSAMP,
        nwave=NWAVE,
        fps=60,
        pitch_estimate=74
    )
    yield cfg


@pytest.fixture(scope="module", params=[n163_cfg, unrounded_cfg])
def read(cfg) -> WaveReader:
    return WaveReader(CFG_DIR, cfg)


if True:
    if False:
        STEREO_PATHS = [Path('test_waves', path) for path in [
            'stereo out of phase.wav',
            'stereo right.wav'
        ]]

        @pytest.mark.parametrize("path", STEREO_PATHS)
        @pytest.mark.parametrize("cfg_factory", [n163_cfg, unrounded_cfg])
        @pytest.fixture(scope="module")
        def stereo_cfg(path, cfg_factory):
            return cfg_factory(
                wav_path=path,
                nsamp=NSAMP,
                nwave=NWAVE,
                fps=60,
                pitch_estimate=74
            )
    else:
        @pytest.fixture(scope="module", params=[
            'stereo out of phase.wav',
            'stereo right.wav'
        ])
        def stereo_path(request):
            path = request.param
            return Path('test_waves', path)


        @pytest.fixture(scope="module", params=[n163_cfg, unrounded_cfg])
        def cfg_factory(request):
            return request.param


        @pytest.fixture(scope="module")
        def stereo_cfg(stereo_path, cfg_factory) -> WaveReaderConfig:
            return cfg_factory(
                wav_path=stereo_path,
                nsamp=NSAMP,
                nwave=NWAVE,
                fps=60,
                pitch_estimate=74
            )

    @pytest.fixture(scope="module")
    def stereo_read(stereo_cfg) -> WaveReader:
        return WaveReader(CFG_DIR, stereo_cfg)

else:
    STEREO_PATHS = [Path('test_waves', path) for path in [
        'stereo out of phase.wav',
        'stereo right.wav'
    ]]

    def stereo_cfg():
        for path in STEREO_PATHS:
            for cfg_factory in [n163_cfg, unrounded_cfg]:
                yield cfg_factory(
                    wav_path=path,
                    nsamp=NSAMP,
                    nwave=NWAVE,
                    fps=60,
                    pitch_estimate=74
                )


    # @pytest.fixture(scope="module")
    def stereo_read() -> Iterable[WaveReader]:
        for cfg in stereo_cfg():
            yield WaveReader(CFG_DIR, cfg)


# Basic tests

def test_wave_at(read):
    """ Ensures wave is not constant. """
    wave = read._wave_at(0)[0]
    assert not (wave == wave[0]).all()


def test_wave_reader(read):
    instr = read.read()
    assert isinstance(instr, Instr)

    waves = instr.waves
    assert len(waves) == NWAVE


# STFT reader tests

def test_read_at(read):
    inds = [0, 1, 3, 6, 10]

    instr = read.read_at(inds)
    assert isinstance(instr, Instr)

    waves = instr.waves
    assert len(waves) == len(inds)

    for i, wave in enumerate(waves):
        assert not (wave == wave[0]).all(), i


def test_reader_instr(read):
    instr = read.read()
    assert isinstance(instr, Instr)

    with io.StringIO() as dummy_stdout, \
            redirect_stdout(dummy_stdout):
        instr.print(2, True)


# Indexing/sweep tests

def test_instr_subset(read):
    instr = read.read()
    sub = instr[:20, 20:10:-1]
    with io.StringIO() as dummy_stdout, \
            redirect_stdout(dummy_stdout):
        sub.print(74)


def test_reader_sweep(cfg):
    cfg = dataclasses.replace(cfg, sweep='0 0 1 1')
    read = WaveReader(CFG_DIR, cfg)
    instr = read.read()

    assert len(instr.waves) == 2
    assert (instr.sweep == [0, 0, 1, 1]).all()


def test_reader_sweep_loop_release(cfg):
    cfg = dataclasses.replace(cfg, sweep='0 / 0 | 1 1')
    read = WaveReader(CFG_DIR, cfg)
    instr = read.read()

    assert len(instr.waves) == 2
    assert (instr.sweep == [0, 0, 1, 1]).all()
    assert instr.sweep.loop == 2
    assert instr.sweep.release == 1


def test_reader_sweep_remove_unused(cfg):
    """ Ensure that waves not present in `sweep` are removed. """
    cfg = dataclasses.replace(cfg, sweep='0 1 3 6 3 1')
    read = WaveReader(CFG_DIR, cfg)
    instr = read.read()

    assert len(instr.waves) == 4
    assert (instr.sweep == [0, 1, 2, 3, 2, 1]).all()


@pytest.mark.xfail(strict=True)
def test_reader_wave_locs(cfg):
    cfg = dataclasses.replace(cfg, wave_locs='0 1 3 6')
    read = WaveReader(CFG_DIR, cfg)
    instr = read.read()

    assert len(instr.waves) == 4
    assert (instr.sweep == map(int, '0 1 1 2 2 2 3'.split())).all()


def test_reader_subsample():
    wave_sub = 2
    env_sub = 3

    # should wave mandatory =N*env?
    for ntick in [3, 4]:
        cfg = n163_cfg(
            wav_path=WAV_PATH,
            nsamp=NSAMP,
            fps=60,
            pitch_estimate=74,

            nwave=ntick,
            wave_sub=wave_sub,
            env_sub=env_sub,
        )
        read = WaveReader(CFG_DIR, cfg)
        instr = read.read()

        # Subsampled waves
        nwave_sub = ceildiv(ntick, wave_sub)
        assert len(instr.waves) == nwave_sub

        # Subsampled sweep
        assert (instr.sweep == np.arange(nwave_sub)).all()

        # Subsampled volume/pitch
        nenv_sub = ceildiv(ntick, env_sub)

        def check(arr: np.ndarray):
            assert len(arr) == nenv_sub

        for i in range(ceildiv(ntick, env_sub)):
            check(instr.vols)
            check(instr.freqs)


# STFT tests

@pytest.mark.parametrize('stft_merge', ['power', 'sum'])
def test_reader_stft_merge(cfg, stft_merge):
    cfg.stft_merge = stft_merge
    read = WaveReader(CFG_DIR, cfg)
    instr = read.read()

    # TODO check instr is legit
    assert instr


def test_cfg_include():
    """Ensure file inclusion works."""
    with pushd('tests'):
        assert recursive_load_yaml('library.n163') == {'library': 1, 'override': 1}
        assert recursive_load_yaml('user.n163') == {'library': 1, 'override': 2, 'user': 3}
        with pytest.raises(ValueError):
            recursive_load_yaml('recursion.n163')


# Stereo tests

def test_reader_stereo(stereo_read):
    instr = stereo_read.read()
    waves = instr.waves

    for i, wave in enumerate(waves):
        assert not (wave == wave[0]).all(), i


from wavetable.wave_reader import yaml

def test_reader_multi_waves():
    cfg_str = '''\

'''
