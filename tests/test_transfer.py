from wavetable.transfers import *


def test_unity():
    unity2 = Unity() * Unity()
    assert unity2(0) == 1
    assert unity2(100) == 1
