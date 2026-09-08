# oracle-capture (Gepard-1.0)

Golden-capture harness for the **Gepard-1.0** Swift-MLX port — Phase 0 of the port plan. It
produces the goldens [`mlx-gepard-swift`](https://github.com/xocialize/mlx-gepard-swift)'s parity
gates assert against.

Brought under version control per **AB-D-0070** (AB-T-0112), which found it tracked by no
repository: the area repo is docs-only, and this harness sat outside every package repo.

## It runs on a REMOTE POD, not here

Unlike the Audio8 harnesses this one is not a local capture. `pod_bootstrap.sh` copies
`capture_goldens.py` + `texts.json` into a `gepard-inference` checkout on a GPU pod, the capture
runs there against the reference implementation, and `fetch_results.sh` pulls the goldens back.
Paths are therefore passed in (`--out /workspace/goldens_out`), not hardcoded — nothing needed
rerooting when the scripts moved here.

```bash
./pod_bootstrap.sh          # provision + upload the kit (incl. extra_ref_audio/*.wav)
# on the pod, from the gepard-inference repo root with its venv active:
python capture_goldens.py --out /workspace/goldens_out
./fetch_results.sh          # bring the goldens home
```

## What it captures

fp32 on CPU by default (the cleanest oracle for the Swift gates): resolved configs and special
ids, tokenizer ids with text-repetition expansion, per-reference packed codec tokens and unfolded
`ref_codes`, the full Q-Former ladder, the fixed-frame audio-interface embedding, prefill hiddens
and 32 head logits plus the stop logit, an uncond+guided CFG pair at w=2.6 that gates the Swift
CFG math, greedy rollout codes with per-step stop probabilities, the split-path codec decode, the
NanoCodec 21.5 fps export that feeds Phase 1, sampled showcase WAVs, and a `manifest.json` with
shape/dtype/sha of every artifact plus env versions.

## `extra_ref_audio/`

A drop folder for candidate reference clips — see its `README.md` for the naming rule (first clip
alphabetically becomes the canonical reference). **The clips themselves are not versioned**: they
are operator-supplied, of the operator's own provenance, and the capture regenerates everything
derived from them. The README is the contract; the audio is not.

## Caveats carried from the port

Gepard decodes greedily and its stop-head truncation is deterministic per (reference, text) —
synthetic bakes trip it, 0.99 rescues, retries never help (AB-L-0075). Seeded sampled rollouts can
never be a parity gate across bindings: `mx.random.normal` after `seed()` is bit-identical
Python↔Swift but `categorical` through the global state is not (AB-L-0090). Parity is asserted on
the greedy path.
