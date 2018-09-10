# Clock rate of NTSC NES/Famicom
from collections import namedtuple

import numpy as np

from wavetable.dsp.wave_util import midi2freq

CPU_NTSC = 1.79e6
FPS = 60
FRAME_TIME = 1.0/FPS

_note = namedtuple('note', 'waves vol pitch')
class Note(_note):
    def length(self):
        return len(self.waves)

    def get(self, i):
        return self.waves[i % self.length()]


class N163Player:
    # 15 cycles per sample, maybe because 0..15 volume levels
    N163_RAW_SR = CPU_NTSC / 15

    def __init__(self, nchan):
        self.nchan = nchan
        self.sr = self.N163_RAW_SR / nchan

    # hungarian but worse
    def s_t(self, time):
        return int(time * self.sr)

    def t_f(self, frame):
        return frame * FRAME_TIME

    def f_t(self, time):
        return int(time / FRAME_TIME)

    def t_s(self, sample):
        return sample / self.sr

    def render(self, note: Note, time=None):
        if time is None:
            time = note.length() * FRAME_TIME

        wavelen = len(note.waves[0])
        freq = midi2freq(note.pitch)

        def idx_phase(phase):
            return (phase * wavelen).astype(int) % wavelen

        def phase_t(t):
            return t * freq

        nsamp = self.s_t(time)
        audio = np.zeros(nsamp)

        t0 = 0

        for f in range(self.f_t(time)):
            sampleBegin = self.s_t(self.t_f(f))
            sampleEnd = self.s_t(self.t_f(f+1))

            wave = note.get(f)

            sample_times = self.t_s(np.arange(sampleBegin, sampleEnd)) - t0
            sample_inds = idx_phase(phase_t(sample_times))
            # print(sample_inds)

            audio[sampleBegin:sampleEnd] = wave[sample_inds]

        audio -= np.mean(audio)

        return audio, round(self.sr)

    def play(self, notes, time=None):
        import IPython.display
        audios = []
        sr = None
        for note in notes:
            wave, sr = self.render(note, time)
            audios.append(wave)

        mixed = sum(audios)

        return IPython.display.Audio(mixed, rate=sr)
