import numpy as np


def ceildiv(n: int, d: int) -> int:
    return -(-n // d)


def seq_along(arr):
    return np.arange(len(arr))
