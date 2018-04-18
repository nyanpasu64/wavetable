## WAV to N163 converter

### Installation:

Install Miniconda3 (<https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>).

```shell
conda install numpy scipy
pip install ruamel.yaml
pip install git+https://github.com/endolith/waveform_analysis.git@4bb2085
```

### Executing

I recommend associating the .n163 file extension with `wave_reader.cmd`, then double-click `config.n163` to run.
Alternatively, you can run `wave_reader.py config.yaml`.

### config.yaml syntax

```yaml
file: "filename.wav"        # quotes are optional
nsamp: N163 wave length
pitch_estimate: 83          # MIDI pitch, middle C4 is 60, C5 is 72.
                                # This tool may estimate the wrong octave, if line is missing.
                                # Exclude if WAV file has pitch changes 1 octave or greater.
at: "0:15 | 15:30 30:15"    # The program generates synchronized wave and volume envelopes. DO NOT EXCEED 0:64 OR 63:0.
                                # 0 1 2 ... 13 14 | 15 16 ... 29 30 29 ... 17 16
                                # TODO: 0:30:10 should produce {0 0 0 1 1 1 ... 9 9 9} (30 items), mimicing FamiTracker behavior.
[optional] nwave: 33        # Truncates output to first `nwave` frames. DO NOT EXCEED 64.
[optional] fps: 240         # Increasing this value will effectively slow the wave down, or transpose the WAV downards. Defaults to 60.
[optional] fft_mode: normal # "zoh" adds a high-frequency boost to compensate for N163 hardware, which may or may not increase high-pitched aliasing sizzle.
```

I recommend using "at" (deletes unused waves) and removing the "nwave" line.

Seamless looping is not supported yet, but bidirectional looping using "at" works very well (since I align adjacent waves to maximize correlation).

### Output Format

j0CC-Famitracker is available at <https://github.com/jimbo1qaz/0CC-FamiTracker/releases.>

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

### Example FDS config.yaml (Audacity may be easier)

```yaml
file: "filename.wav"
nsamp: 64
range: 64
vol_range: 33
pitch_estimate: 83              # MIDI pitch, middle C4 is 60, C5 is 72.
at: 10                          # Pick time (in units of 1/60th second)
[optional] fft_mode: normal     # "zoh" adds a slight high-frequency boost to compensate for FDS hardware.
```

## N163 Instrument Merger

There is no CLI available at the moment. I suggest placing this folder with PYTHONPATH and using IPython notebooks or Python files.
