import io

import dataclasses
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from wavetable.instrument import Instr
from wavetable.util.math import ceildiv
from wavetable.wave_reader import WaveReader, n163_cfg, unrounded_cfg, WaveReaderConfig


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
def read(request) -> WaveReader:
    """ request.param is a cfg factory, taken from params.
    "request" is a hardcoded name. """
    cfg = request.param(
        wav_path=WAV_PATH,
        nsamp=NSAMP,
        nwave=NWAVE,
        fps=60,
        pitch_estimate=74
    )
    yield WaveReader(CFG_DIR, cfg)


def test_wave_at(read):
    """ Ensures wave is not constant. """
    wave = read._wave_at(0)[0]
    assert not (wave == wave[0]).all()


def test_wave_reader(read):
    instr = read.read()
    assert isinstance(instr, Instr)

    waves = instr.waves
    assert len(waves) == NWAVE


def test_read_at(read):
    inds = [0, 1, 3, 6, 10]

    instr = read.read_at(inds)
    assert isinstance(instr, Instr)

    waves = instr.waves
    assert len(waves) == len(inds)


def test_reader_instr(read):
    instr = read.read()
    assert isinstance(instr, Instr)

    with io.StringIO() as dummy_stdout, \
            redirect_stdout(dummy_stdout):
        instr.print(2, True)


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
