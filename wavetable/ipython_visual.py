import matplotlib.pyplot as plt
import numpy as np

# https://hackernoon.com/log-x-vs-ln-x-the-curse-of-scientific-computing-170c8e95310c
np.loge = np.ln = np.log


def reassign_ticks(dy):
    ymin, ymax = plt.ylim()
    ymin = int(ymin / dy) * dy
    yticks = np.arange(ymin, ymax, dy)
    plt.yticks(yticks)


def spectrogram(read):
    fs, ts, Zxx = read.stft
    # Zxx[freq][time]
    # Zxx[y↑][x→]

    plt.figure(figsize=(20, 10))
    plt.pcolormesh(ts, fs, np.log(np.abs(Zxx)))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    print(len(ts))
    print(len(fs))
    print(Zxx.shape)


def _spectrum(fft):
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    #     print(fs)
    #     print(ts)
    fft = 20 * np.log10(abs(fft))
    plt.plot(fft)

    reassign_ticks(6)


def spectrum(read, i):
    fs, ts, Zxx = read.stft
    # Zxx[freq][time]
    # Zxx[y↑][x→]

    _spectrum(Zxx[..., i])
