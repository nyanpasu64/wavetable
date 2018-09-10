import io
from contextlib import redirect_stdout
from pathlib import Path

import dataclasses
import numpy as np
import pytest

from wavetable.dsp.fourier import rfft_norm
from wavetable.types.instrument import Instr
from wavetable.util.fs import pushd
from wavetable.util.math import ceildiv
from wavetable.wave_reader import WaveReader, n163_cfg, unrounded_cfg, WaveReaderConfig, \
    recursive_load_yaml
from wavetable.wave_reader import yaml


CFG_DIR = Path('tests')
WAVE_DIR = CFG_DIR / 'test_waves'
WAV_PATH = Path('test_waves/Sample 19.wav')
NSAMP = 128
NWAVE = 30


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


SINE = '''\
wav_path: sine440.69.wav
nwave: 1
nsamp: 16
'''


def test_reader_pitch_metadata():
    """ Ensure WaveReader obtains root pitch from wav filename. """
    cfg_dict = yaml.load(SINE)
    # root_pitch should be obtained from filename
    read = WaveReader(WAVE_DIR, n163_cfg(**cfg_dict))
    read.read()

    cfg_dict['wav_path'] = 'Sample 19.wav'
    with pytest.raises(TypeError):
        read = WaveReader(WAVE_DIR, n163_cfg(**cfg_dict))
        read.read()


# Phase-assignment tests

def get_phase(cfg):
    read = WaveReader(WAVE_DIR, cfg)
    instr = read.read()

    spectrum = rfft_norm(instr.waves[0])
    phase = np.angle(spectrum[1])
    return phase


def test_reader_phase_f():
    """ Ensure phase_f=lambda works. """
    cfg = cfg_yaml(SINE)
    tolerance = 0.01

    assert abs(get_phase(cfg)) > tolerance

    cfg.phase_f = 'lambda f: 0'
    assert abs(get_phase(cfg)) < tolerance


def test_reader_phase():
    """ Ensure phase=float works. """
    cfg = cfg_yaml(SINE)
    tolerance = 0.01

    cfg.phase = 0
    assert abs(get_phase(cfg)) < tolerance


# Stereo tests

def test_reader_stereo(stereo_read):
    instr = stereo_read.read()
    waves = instr.waves

    for i, wave in enumerate(waves):
        assert not (wave == wave[0]).all(), i


def test_reader_multi_waves():
    cfg_str = '''\
files:
  - path: sine440.69.wav
  - path: sine440.69.wav
    speed: 2
    volume: 1.1
  - path: sine256.59.624.wav
    speed: 3
    volume: 0
pitch_estimate: 69
nwave: 1

nsamp: 16
'''
    harmonics = [1, 2]  # plus 3 at volume=0

    cfg = n163_cfg(**yaml.load(cfg_str))
    read = WaveReader(WAVE_DIR, cfg)
    instr = read.read()

    # Calculate spectrum of resulting signal
    spectrum: np.ndarray = np.abs(rfft_norm(instr.waves[0]))
    spectrum[0] = 0
    threshold = np.amax(spectrum) / 2
    assert threshold > 0.1

    # Ensure pitches present
    assert (spectrum[harmonics] > threshold).all()

    # Ensure no other pitches present
    spectrum[harmonics] = 0
    spectrum[0] = 0
    assert (spectrum < threshold).all()


# Fixtures

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


def cfg_yaml(yaml_str):
    return n163_cfg(**yaml.load(yaml_str))
