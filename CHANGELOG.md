# Changelog

???:

- wave_reader (.n163):
    - Replace `width_frames` with `segment_ms`
        - STFT time changed from  `1/fps * width_frames` to `segment_time`
    - Add optional parameters `wave_sub`, `env_sub` to `fps`
        - Subsample waves or envelopes to a rate of `fps/sub`
        - `wave_sub` controls waves and wave indices
        - `env_sub` controls volume and frequency
        - STFT computational load decreased by factor of `gcd(*foo_sub)`
    - Deprecate `at`.
    <!-- - `fps` deprecated by `wave_fps` and `_fps`
        - `wave_fps` controls waves and volume???
        - `pitch_fps` controls pitch resolution -->
