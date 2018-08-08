import numpy as np
import pytest

from wavetable.instrument import Instr
from wavetable.wave_util import A


NSAMP = 16
WAVES = np.array([
    np.linspace(0, NSAMP - 1, NSAMP),
    np.linspace(NSAMP - 1, 0, NSAMP)
])

NWAVE = 2

@pytest.fixture
def instr() -> Instr:
    return Instr(
        WAVES,
        sweep=None,
        vols=np.arange(NWAVE),
        freqs=np.arange(NWAVE)
    )


def test_instr_index(instr):
    """ Verify Instr.__getitem__() indexing works properly.
    It's intended to slice all seqs, but leave waves untouched unless
    you call remove_unused_waves().
    """

    new = instr[[1]]
    assert new == Instr(
        WAVES,
        sweep=[1],
        vols=[1],
        freqs=[1]
    )

    new.remove_unused_waves()
    assert new == Instr(
        WAVES[[1]],
        sweep=[0],
        vols=[1],
        freqs=[1]
    )

