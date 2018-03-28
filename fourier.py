from typing import List, Union

import numpy as np

def irfft(spectrum: Union[np.ndarray, List[complex]], nout=None):
    """
    Calculate the inverse Fourier transform of $spectrum, optionally
    truncating to $nout entries.

    if nout is None, calculate nout = 2*(len-1).
        nin = (nout//2) + 1.

    if nout is even, realify the last coeff to preserve Nyquist energy.
    """
    nout_orig = nout

    if nout is None:
        nout = 2 * (len(spectrum) - 1)
    nin = nout // 2 + 1

    if nout_orig is None:
        assert nin == len(spectrum)

    spectrum = spectrum[:nin].copy()
    if nout % 2 == 0:
        last = spectrum[-1]
        spectrum[-1] = np.copysign(abs(last), last.real)

    return np.fft.irfft(spectrum, nout)


