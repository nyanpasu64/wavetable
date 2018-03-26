from wavetable.merge import *
import numpy as np


def test_merge_waves_mml():
    waves = load_string('0 1 2 3; 0 1 3 6;')
    mml = '0 1 0 1'
    vol_curve = '0 1 2 3 4 3 2 1 0'
    assert (merge_waves_mml(waves, mml) ==
            np.array([[0, 1, 2, 3],
                      [0, 1, 3, 6],
                      [0, 1, 2, 3],
                      [0, 1, 3, 6]])
            ).all()

    assert (merge_waves_mml(waves, mml, vol_curve) ==
            np.array([[0., 0., 0., 0.],
                      [0., 1., 3., 6.],
                      [0., 2., 4., 6.],
                      [0., 3., 9., 18.],
                      [0., 4., 12., 24.],
                      [0., 3., 9., 18.],
                      [0., 2., 6., 12.],
                      [0., 1., 3., 6.],
                      [0., 0., 0., 0.]])
            ).all()

    short_vol = '4 2 1'
    assert (merge_waves_mml(waves, mml, short_vol) ==
            np.array([[0., 4., 8., 12.],
                      [0., 2., 6., 12.],
                      [0., 1., 2., 3.],
                      [0., 1., 3., 6.]])
            ).all()


def test_correlate():
    fixed = F('0 0 1 2 0 0')
    sweep = F('0 0 1 2 0 0')
    N = len(sweep)

    assert correlate_offset(fixed, sweep) == 0

    sweep = F('0 0 0 1 2 0')
    offset = correlate_offset(fixed, sweep)
    assert offset == N - 1

    assert (np.roll(sweep, offset) == fixed).all()


if __name__ == '__main__':
    import py

    py.test.cmdline.main(sys.argv)
