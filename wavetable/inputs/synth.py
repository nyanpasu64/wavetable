import numpy as np

def roll_speed_offset(N, wave, a, b):
    waves = np.array([np.roll(wave, a * i + b)
             for i in range(N)])
    return waves
