# Drop Roxy candidate reference clips here (before uploading the kit to the pod)

- `.wav` files, 5–15 s of clean single-speaker speech (any sample rate — the capture
  resamples to 22.05 kHz mono itself, and saves the exact resampled wave as a golden).
- 2–3 candidates is ideal: the capture generates cloned showcase WAVs for **every**
  clip in this folder, so the pod session doubles as the Roxy voice audition.
- The **first clip alphabetically** becomes the canonical ref used for the Q-Former /
  prefill / rollout goldens — name your leading candidate accordingly (e.g.
  `roxy_a.wav`).
- Generate candidates with VoxCPM2 / Qwen3-TTS (the fleet's `.tts` providers), or use
  any clip you have rights to.

If this folder has no wavs, the capture falls back to gepard-inference's demo clips
(`ref_audio/audio_en.wav` etc.) — goldens still work, but you lose the audition.
