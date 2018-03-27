import py.test
import pytest
from wavetable.instrument import Instr

from util import AttrDict
from wave_reader import WaveReader

assert py.test

PATH = 'test_waves/Sample 19.wav'
FPS = 60
MAX_RANGE = 16
NSAMP = 128
NWAVE = 30

cfg = AttrDict(
    range=MAX_RANGE,
    nsamp=NSAMP,
    nwave=NWAVE,
    fps=60,
    pitch_estimate=74
)


def test_wave_at():
    read = WaveReader(PATH, cfg)
    wave = read.wave_at(0)
    assert not (wave == wave[0]).all()


def test_wave_reader():
    read = WaveReader(PATH, cfg)
    instr = read.read()
    assert isinstance(instr, Instr)

    waveseq = instr.waveseq
    assert len(waveseq) == NWAVE


def test_read_at():
    inds = [0, 1, 3, 6, 10]

    read = WaveReader(PATH, cfg)
    instr = read.read_at(inds)
    assert isinstance(instr, Instr)

    waveseq = instr.waveseq
    assert len(waveseq) == len(inds)
