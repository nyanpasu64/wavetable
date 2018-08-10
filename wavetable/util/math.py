import numpy as np


def ceildiv(n: int, d: int) -> int:
    return -(-n // d)


def seq_along(arr):
    return np.arange(len(arr))


def nearest_sub_harmonic(precise: float, accurate: float) -> float:
    """ Finds the nearest sub/harmonic of `precise` to `accurate`. """
    if precise > accurate:
        precise /= round(precise / accurate)
    elif precise < accurate:
        precise *= round(accurate / precise)
    return precise
