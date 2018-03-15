# Clock rate of NTSC NES/Famicom
from collections import namedtuple

CPU_NTSC = 1.79e6
FPS = 60
FRAME_TIME = 1.0/FPS


def pitch2freq(pitch):
    freq = 440 * 2 ** ((pitch - 9) / 12)
    return freq


_note = namedtuple('note', 'waveseq vol pitch')
class Note(_note):
    def length(self):
        return len(self.waveseq)

    def get(self, i):
        return self.waveseq[i % self.length()]


class N163Player:
    # 15 cycles per sample, maybe because 0..15 volume levels
    N163_RAW_SR = CPU_NTSC / 15

    def __init__(self, nchan):
        self.nchan = nchan
        self.sr = self.N163_RAW_SR / nchan

    def play(self, note: Note, time=None):
        if time is None:
            time = note.length() * FRAME_TIME
            freq = pitch2freq(note.pitch)



