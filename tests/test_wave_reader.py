import io
from contextlib import redirect_stdout
from pathlib import Path

import py.test
import pytest
from wavetable.instrument import Instr

from wavetable.wave_reader import WaveReader, n163_cfg, unrounded_cfg

assert py.test

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

    waveseq = instr.waveseq
    assert len(waveseq) == NWAVE


def test_read_at(read):
    inds = [0, 1, 3, 6, 10]

    instr = read.read_at(inds)
    assert isinstance(instr, Instr)

    waveseq = instr.waveseq
    assert len(waveseq) == len(inds)


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
