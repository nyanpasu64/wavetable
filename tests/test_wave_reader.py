import io
import numpy as np
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from wavetable.instrument import Instr
from wavetable.util.math import ceildiv
from wavetable.wave_reader import WaveReader, n163_cfg, unrounded_cfg


CFG_DIR = Path('tests')
WAV_PATH = Path('test_waves/Sample 19.wav')
NSAMP = 128
NWAVE = 30


@pytest.fixture(scope="module", params=[n163_cfg, unrounded_cfg])
def read(request):
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


def test_subset(read):
    instr = read.read()
    sub = instr[:20, 20:10:-1]
    with io.StringIO() as dummy_stdout, \
            redirect_stdout(dummy_stdout):
        sub.print(74)


def test_subsample():
    wave_sub = 2
    env_sub = 3

    # should wave mandatory =N*env?
    for nwave in [3, 4]:    # , 5]:
        cfg = n163_cfg(
            wav_path=WAV_PATH,
            nsamp=NSAMP,
            fps=60,
            pitch_estimate=74,

            nwave=nwave,
            wave_sub=wave_sub,
            env_sub=env_sub,
        )
        read = WaveReader(CFG_DIR, cfg)
        instr = read.read()

        # Subsampled waves
        assert len(instr.waves) == ceildiv(nwave, wave_sub)

        # Subsampled sweep
        sweep = [i // wave_sub for i in range(nwave)]
        assert (instr.sweep == sweep).all()

        # Subsampled volume/pitch
        def check(arr: np.ndarray):
            assert (arr[begin:end] == arr[begin]).all()

        for i in range(ceildiv(nwave, env_sub)):
            begin = env_sub * i
            end = env_sub * (i + 1)
            check(instr.vols)
            check(instr.freqs)
