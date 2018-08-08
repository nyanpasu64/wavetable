# Wavetable API (N163, SNES BRR)

## Installation

Install Python 3.6 (earlier versions are unsupported), and run the following commands:

```shell
pip install -e .
```

(I used to recommend Miniconda3 `conda install numpy scipy`, but pip now works on Windows, without a compiler and inordinate build times.)

## N163 Ripper

I recommend associating the .n163 file extension with `wave_reader.cmd`, then double-click `config.n163` to run.
Alternatively, you can run `wave_reader.py config.yaml`.

### config.yaml syntax

```yaml
wav_path: "filename.wav"    # quotes are optional
nsamp: N163 wave length
pitch_estimate: 83          # MIDI pitch, middle C4 is 60, C5 is 72.
                                # This tool may estimate the wrong octave, if line is missing.
                                # (fixme crashes) Exclude if WAV file has pitch changes 1 octave or greater.

# All below are optional.

fps: 240         # Increasing this value will effectively slow the wave down, or transpose the WAV downards. Defaults to 60.
wave_sub: 2      # Subsampling factor for waves (and wave indices)
env_sub: 1       # Subsampling factor for volume and frequency

(TODO) wave_locs: 0 1 3 6              # Rip four waves at frames 0,1,3,6, and generate wave envelope 0 1 1 2 2 2 3.
sweep: "0:15 | 15:30 30:15"     # Generates synchronized wave and volume envelopes.
nwave: 33        # Truncates output to first `nwave` frames. DO NOT EXCEED 64.
fft_mode: normal # "zoh" adds a high-frequency boost to compensate for N163 hardware, which may or may not increase high-pitched aliasing sizzle.
```

- foo is subsampled, and each entry is repeated by a factor of `foo_sub`.
- STFT computational load decreased by factor of `gcd(*foo_sub)`.

I recommend using "at" (deletes unused waves) and removing the "nwave" line.

Seamless looping is not supported yet, but bidirectional looping using "at" works very well (since I align adjacent waves to maximize correlation).

### Output Format

j0CC-Famitracker is available at <https://github.com/nyanpasu64/j0CC-FamiTracker/releases>.

```text
Ripped Waves (copy to clipboard, click Paste button in j0CC-Famitracker instrument editor):
0 0 0 2 4 7 9 10 11 12 14 15 15 14 12 12 13 13 12 8 5 4 5 6 7 7 7 7 5 4 2 1; ...
1 0 0 1 3 6 9 11 13 13 14 15 15 15 14 14 15 14 12 9 6 5 6 6 6 5 4 5 5 5 4 3

Wave Index
0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 16 17 ...

vols:
15 11 6 5 5 7 8 6 | 6 6 9 8 9 9 12 11 8 9 ...

pitch (unaffected by "at"):
-13 -14 -14 -15 -14 -11 -12 -15 -22 -16 -20 -16 -13 -16 -15 -12 -14 -16 -13 -14 -17 -16 -9 -13
```

### Example FDS config.yaml (better frequency response than Audacity)

```yaml
file: "filename.wav"
nsamp: 64
range: 64
vol_range: 33
pitch_estimate: 83              # MIDI pitch, middle C4 is 60, C5 is 72.
at: 10                          # Pick time (in units of 1/60th second)
[optional] fft_mode: normal     # "zoh" adds a slight high-frequency boost to compensate for FDS hardware.
```

## SNES BRR wavetable ripper

Generates BRR files, swap the loop point, instrument index, sth.

### config.wtbrr syntax

All SPC-specific parameters are optional. `unlooped_prefix` should be a multiple of 16, recommended equal to `nsamp`. All other parameters should not be supplied, except for testing.

```yaml
no_brr: False           # If True, only generates WAV files and not BRR.
unlooped_prefix: 0      # Controls the loop point of the wave.
truncate_prefix: True   # Remove unlooped prefix from non-initial samples.
    # The prefix will never be used.
gaussian: True          # Pre-emphasis filter to counteract SNES Gaussian filter
```

## N163 Instrument Merger

There is no CLI available at the moment. I suggest placing this folder with PYTHONPATH and using IPython notebooks or Python files.
