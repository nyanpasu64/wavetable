import py.test
import pytest
from wavetable.instrument import Instr

from wavetable.wave_reader import WaveReader, n163_cfg, unrounded_cfg

assert py.test

PATH = 'test_waves/Sample 19.wav'
NSAMP = 128
NWAVE = 30


@pytest.fixture(scope="module", params=[n163_cfg, unrounded_cfg])
def cfg(request):
    """ request.param is a cfg factory, taken from params.
    "request" is a hardcoded name. """
    yield request.param(
        nsamp=NSAMP,
        nwave=NWAVE,
        fps=60,
        pitch_estimate=74
    )


def test_wave_at(cfg):
    """ Ensures wave is not constant. """
    read = WaveReader(PATH, cfg)
    wave = read.wave_at(0)[0]
    assert not (wave == wave[0]).all()


def test_wave_reader(cfg):
    read = WaveReader(PATH, cfg)
    instr = read.read()
    assert isinstance(instr, Instr)

    waveseq = instr.waveseq
    assert len(waveseq) == NWAVE


def test_read_at(cfg):
    inds = [0, 1, 3, 6, 10]

    read = WaveReader(PATH, cfg)
    instr = read.read_at(inds)
    assert isinstance(instr, Instr)

    waveseq = instr.waveseq
    assert len(waveseq) == len(inds)


def test_reader_instr(cfg):
    read = WaveReader(PATH, cfg)
    instr = read.read()
    assert isinstance(instr, Instr)

    instr.print(2, True)


def test_subset(cfg):
    read = WaveReader(PATH, cfg)
    instr = read.read()
    sub = instr[:20, 20:10:-1]
    sub.print(74)
