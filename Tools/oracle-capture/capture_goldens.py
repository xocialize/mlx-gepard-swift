#!/usr/bin/env python3
"""Gepard-1.0 golden-capture harness for the Swift-MLX port (Phase 0 of
mlxengine-audio/Docs/Gepard-Swift-Port-Plan.md).

Run FROM THE gepard-inference REPO ROOT with its venv active (pod_bootstrap.sh
copies this file + texts.json there):

    python capture_goldens.py --out /workspace/goldens_out

Captures, in fp32 on CPU by default (cleanest oracle for the Swift parity gates):
  configs/           gepard_config.json + resolved backbone config + special ids
  token_ids.json     per-text tokenizer ids + text-repetition expansion + R
  refs/<clip>/       exact 22.05k mono wave, packed codec tokens, unfolded ref_codes
  qformer/           input_proj / pos_enc / per-block / q_normed / prefix_out ladder
  audio_interface/   fixed-frame audio embedding golden
  prefill/           layer0 + last-layer hiddens, last_hidden_state, 32 head logits, stop logit
  cfg/               uncond prefill + guided (w=2.6) frame-0 logits — gates Swift CFG math
  rollout/<text>/    greedy (argmax) codes [32,T] + per-step stop probs
  codec/             dequantized decoder input + decoded waveform golden (split-path asserted)
  codec_export/      NanoCodec 21.5fps state_dict -> safetensors + config.yaml  (Phase-1 input)
  showcase/          sampled WAVs per ref clip (temp 0.3, seeded) incl. CFG short-word rescue
  manifest.json      shape/dtype/sha of every artifact + env versions + geometry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# gepard-inference package (run from its repo root)
from gepard_inference.runner import GepardRunner
from gepard_inference.codec_wrapper import Player
from gepard_inference.codec_ops import dequantize_codes

# ----------------------------------------------------------------------------


def sha16(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


class Saver:
    """Writes .npy artifacts under root/ and records shape/dtype/sha in a manifest."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = {"artifacts": {}, "values": {}, "env": {}}

    def npy(self, name: str, arr) -> None:
        arr = np.asarray(arr)
        p = self.root / f"{name}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, arr)
        self.manifest["artifacts"][name] = {
            "shape": list(arr.shape), "dtype": str(arr.dtype), "sha256_16": sha16(arr),
        }
        print(f"  [saved] {name}.npy  {list(arr.shape)} {arr.dtype}")

    def tensor(self, name: str, t: torch.Tensor) -> None:
        t = t.detach().cpu()
        self.npy(name, t.to(torch.float32).numpy() if t.is_floating_point() else t.numpy())

    def value(self, name: str, v) -> None:
        self.manifest["values"][name] = v

    def text(self, name: str, s: str) -> None:
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(s)
        print(f"  [saved] {name}")

    def wav(self, name: str, wave: np.ndarray, sr: int) -> None:
        p = self.root / f"{name}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, wave, sr)
        dur = len(wave) / sr
        print(f"  [saved] {name}.wav  {dur:.2f}s @ {sr}Hz  peak {np.abs(wave).max():.3f}")


def hook_capture(module: torch.nn.Module, store: dict, key: str):
    """Forward hook capturing a module's output (first element if tuple)."""
    def fn(_m, _inp, out):
        store[key] = (out[0] if isinstance(out, tuple) else out).detach()
    return module.register_forward_hook(fn)


def argmax_frame(runner: GepardRunner, h: torch.Tensor) -> torch.Tensor:
    """Deterministic frame: argmax per codebook head, mirroring _sample_frame's head order."""
    toks = []
    for head in runner.model.codebook_heads:
        toks.append(head(h).float().argmax(dim=-1).squeeze())
    return torch.stack(toks)  # (num_heads,)


def greedy_rollout(runner: GepardRunner, prefix, cond_ids, max_frames: int):
    """Replicates GepardRunner.generate() with argmax sampling (single-pass, no CFG).

    Mirrors the reference loop exactly: frame 0 is sampled from the prefill's last
    position with NO stop check; each later step embeds the previous frame, runs one
    decode step (kv_len = K + T_text + step), checks the stop head BEFORE sampling.
    """
    hidden, cache, K, T_text = runner._prefill(prefix, cond_ids)
    frames = [argmax_frame(runner, hidden[:, -1, :])]
    stop_probs = []
    for step in range(1, max_frames):
        frame_embed = runner._embed_frame(frames[-1].unsqueeze(0))
        hidden, cache = runner._decode_step(frame_embed, cache, K + T_text + step)
        stop_logit = runner.model.stop_head(hidden[:, -1, :])
        p = torch.sigmoid(stop_logit.squeeze()).item()
        stop_probs.append(p)
        if p > 0.5:
            break
        frames.append(argmax_frame(runner, hidden[:, -1, :]))
    tokens = torch.stack(frames, dim=0).T.contiguous()  # (num_heads, T)
    return tokens, np.asarray(stop_probs, dtype=np.float32), K, T_text


# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="nineninesix/gepard-1.0")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; cleanest oracle) or cuda (rescue for slow pods)")
    ap.add_argument("--out", default="goldens_out")
    ap.add_argument("--texts", default="texts.json")
    ap.add_argument("--rollout-frames", type=int, default=120)
    ap.add_argument("--skip-showcase", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    out = Saver(Path(args.out))
    dev = torch.device(args.device)

    # -- env snapshot ---------------------------------------------------------
    import transformers
    try:
        import nemo
        nemo_ver = nemo.__version__
    except Exception:
        nemo_ver = "unknown"
    import librosa
    out.manifest["env"] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "nemo": nemo_ver, "numpy": np.__version__, "librosa": librosa.__version__,
        "device": args.device, "dtype": "float32",
        "checkpoint": args.checkpoint,
    }
    print(f"env: {out.manifest['env']}")

    # -- load model + codec, force fp32 --------------------------------------
    print("\n=== load GepardRunner (this downloads ~1.1GB on first run) ===")
    runner = GepardRunner.from_checkpoint(args.checkpoint, device=args.device,
                                          attn_implementation="eager")
    runner.model.float().eval()

    print("\n=== load codec Player (NeMo import is slow, ~1 min) ===")
    player = Player.from_checkpoint(args.checkpoint, device=args.device)
    player.codec.float().eval()
    sr = player.sample_rate
    fsq_levels = list(player.fsq_levels)          # [8, 7, 6, 6]
    num_layers = 8                                 # groups; asserted below
    vq = player.codec.vector_quantizer
    assert int(vq.num_groups) == num_layers, f"num_groups={vq.num_groups}"
    out.value("fsq_levels", fsq_levels)
    out.value("codec_num_groups", int(vq.num_groups))
    out.value("sample_rate", sr)
    out.value("vocab_sizes", list(runner.model.vocab_sizes))
    out.value("special_tokens", {"BOS_TEXT": runner.BOS_TEXT, "EOT": runner.EOT,
                                 "BOS_AUDIO": runner.BOS_AUDIO})

    # -- configs --------------------------------------------------------------
    print("\n=== configs ===")
    from huggingface_hub import hf_hub_download
    import shutil
    (out.root / "configs").mkdir(exist_ok=True)
    for fname in ("gepard_config.json", "config.json", "tokenizer_config.json"):
        try:
            shutil.copy(hf_hub_download(args.checkpoint, fname), out.root / "configs" / fname)
            print(f"  [saved] configs/{fname}")
        except Exception as e:  # tokenizer_config may not matter; don't die
            print(f"  [warn] {fname}: {e}")
    out.text("configs/backbone_resolved.json",
             json.dumps(runner.model.config.to_dict(), indent=2, default=str))

    # -- texts: ids + repetition expansion ------------------------------------
    print("\n=== texts / tokenizer ===")
    texts = json.loads(Path(args.texts).read_text())
    tok_info = {}
    for name, text in texts.items():
        ids = runner.tokenizer.encode(text, add_special_tokens=False)
        R = runner.repeater.target_R(len(ids))
        cond_ids = runner.repeater.expand(ids)
        tok_info[name] = {"text": text, "ids": ids, "R": R, "cond_ids": cond_ids}
        print(f"  {name}: {len(ids)} tokens, R={R}, cond_len={len(cond_ids)}")
    out.text("token_ids.json", json.dumps(tok_info, indent=2))

    # -- reference clips: wave -> packed tokens -> unfolded ref_codes ---------
    print("\n=== reference clips ===")
    ref_paths = sorted(Path("extra_ref_audio").glob("*.wav")) + sorted(Path("ref_audio").glob("*.wav"))
    if not ref_paths:
        sys.exit("no reference clips found (ref_audio/ or extra_ref_audio/)")
    refs = {}
    for path in ref_paths:
        stem = path.stem
        wave_np, _ = librosa.load(str(path), sr=sr, mono=True)
        out.npy(f"refs/{stem}/wave_{sr}", wave_np.astype(np.float32))
        wave = torch.from_numpy(wave_np).unsqueeze(0).to(dev)
        wave_len = torch.tensor([wave.shape[-1]], device=dev)
        with torch.inference_mode():
            packed, _ = player.codec.encode(audio=wave, audio_len=wave_len)  # [1, C, T]
        out.tensor(f"refs/{stem}/packed_tokens", packed)
        ref_codes = player.encode_reference(str(path))                        # [1, T, C_total]
        out.tensor(f"refs/{stem}/ref_codes", ref_codes)
        refs[stem] = ref_codes
    # canonical ref = first extra clip if present (the Roxy candidate), else demo clip
    canonical_ref_name = ref_paths[0].stem
    out.value("canonical_ref", canonical_ref_name)
    ref_codes = refs[canonical_ref_name].to(dev)
    print(f"  canonical ref clip: {canonical_ref_name}")

    # -- Q-Former ladder -------------------------------------------------------
    print("\n=== Q-Former (RefCompressor) ladder ===")
    rc = runner.model.ref_compressor
    assert rc is not None
    caps: dict = {}
    hooks = [hook_capture(rc.input_proj, caps, "input_proj_out"),
             hook_capture(rc.pos_enc, caps, "pos_enc_out")]
    hooks += [hook_capture(b, caps, f"block{i}_out") for i, b in enumerate(rc.blocks)]
    ref_mask = torch.ones(ref_codes.shape[0], ref_codes.shape[1], dtype=torch.bool, device=dev)
    with torch.inference_mode():
        prefix_out, q_normed = rc(ref_codes, ref_mask)
    for h in hooks:
        h.remove()
    for k, v in caps.items():
        out.tensor(f"qformer/{k}", v)
    out.tensor("qformer/prefix_out", prefix_out)
    out.tensor("qformer/q_normed", q_normed)

    # -- audio interface golden (fixed deterministic frame) --------------------
    print("\n=== audio interface ===")
    vocab = runner.model.vocab_sizes
    fixed = torch.tensor([(i * 5 + 3) % vocab[i] for i in range(len(vocab))],
                         dtype=torch.long, device=dev)
    out.tensor("audio_interface/fixed_frame_tokens", fixed)
    with torch.inference_mode():
        emb = runner._embed_frame(fixed.unsqueeze(0))  # (1, 1, d)
    out.tensor("audio_interface/fixed_frame_embed", emb)
    out.tensor("audio_interface/audio_embed_scale", runner.model.audio_embed_scale)

    # -- prefill golden (canonical text + canonical prefix) --------------------
    print("\n=== prefill (canonical) ===")
    cond_ids = tok_info["canonical"]["cond_ids"]
    ids_t = torch.tensor([tok_info["canonical"]["ids"]], dtype=torch.long, device=dev)
    with torch.inference_mode():
        out.tensor("prefill/text_embeds_raw_ids", runner.model.model.embed_tokens(ids_t))
        prefix = runner._compute_ref_prefix(ref_codes, None)
        assert torch.allclose(prefix, prefix_out, atol=0), "prefix mismatch vs qformer capture"
        layer_caps: dict = {}
        layers = runner.model.model.layers
        lh = [hook_capture(layers[0], layer_caps, "layer0_out"),
              hook_capture(layers[-1], layer_caps, "layer_last_out")]
        hidden, _cache, K, T_text = runner._prefill(prefix, cond_ids)
        for h in lh:
            h.remove()
    out.tensor("prefill/last_hidden", hidden)
    out.tensor("prefill/layer0_out", layer_caps["layer0_out"])
    out.tensor("prefill/layer_last_out", layer_caps["layer_last_out"])
    out.value("prefill_K", int(K))
    out.value("prefill_T_text", int(T_text))
    h_last = hidden[:, -1, :]
    with torch.inference_mode():
        for i, head in enumerate(runner.model.codebook_heads):
            out.tensor(f"prefill/head{i:02d}_logits", head(h_last).float())
        out.tensor("prefill/stop_logit", runner.model.stop_head(h_last).float())

    # -- CFG golden: guided frame-0 logits (w=2.6) on the 1-word text ----------
    print("\n=== CFG golden (one_word, w=2.6) ===")
    W = 2.6
    with torch.inference_mode():
        cond_hidden, _, _, _ = runner._prefill(prefix, tok_info["one_word"]["cond_ids"])
        uncond_ids = [runner.BOS_TEXT, runner.EOT, runner.BOS_AUDIO]
        unc_hidden, _, _, _ = runner._prefill(prefix, uncond_ids)
        hc, hu = cond_hidden[:, -1, :], unc_hidden[:, -1, :]
        out.tensor("cfg/uncond_last_hidden", unc_hidden)
        for i, head in enumerate(runner.model.codebook_heads):
            lc, lu = head(hc).float(), head(hu).float()
            out.tensor(f"cfg/head{i:02d}_guided_logits", lu + W * (lc - lu))
    out.value("cfg_scale", W)
    out.value("cfg_uncond_ids", uncond_ids)

    # -- greedy rollouts --------------------------------------------------------
    for name in ("canonical", "one_word"):
        print(f"\n=== greedy rollout: {name} (<= {args.rollout_frames} frames) ===")
        with torch.inference_mode():
            tokens, stop_probs, _, _ = greedy_rollout(
                runner, prefix, tok_info[name]["cond_ids"], args.rollout_frames)
        out.tensor(f"rollout/{name}/codes", tokens)
        out.npy(f"rollout/{name}/stop_probs", stop_probs)
        print(f"  {tokens.shape[1]} frames ({tokens.shape[1] / 21.5:.2f}s), "
              f"stopped={'yes' if len(stop_probs) and stop_probs[-1] > 0.5 else 'cap'}")

    # -- codec decode golden (+ split-path assert) ------------------------------
    print("\n=== codec decode ===")
    codes = torch.from_numpy(
        np.load(out.root / "rollout/canonical/codes.npy")).to(dev)  # (32, T)
    with torch.inference_mode():
        sr_dec, wave_ref = player.decode(codes)
    assert sr_dec == sr
    out.npy("codec/decoded_wave", wave_ref.astype(np.float32))
    out.wav("codec/decoded_wave", wave_ref, sr)
    # split-path: our own dequantize (the op the Swift port implements) must
    # reproduce decode_from_codes' internal dequantization exactly.
    with torch.inference_mode():
        deq = dequantize_codes(codes.T.unsqueeze(0), fsq_levels, num_layers)  # [1, T, 32]
        out.tensor("codec/dequantized_input", deq)
        audio2, _ = player.codec.decode_audio(
            inputs=deq.permute(0, 2, 1).contiguous(),
            input_len=torch.tensor([codes.shape[1]], device=dev))
    wave_split = audio2.float().cpu().flatten().numpy()
    assert np.array_equal(wave_split, wave_ref), "split-path decode mismatch!"
    print("  split-path dequantize→decode_audio == decode_from_codes ✓")

    # -- codec export: weights + config for the Phase-1 nanocodec-mlx adaptation
    print("\n=== codec export (safetensors + yaml) ===")
    from safetensors.torch import save_file
    (out.root / "codec_export").mkdir(exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in player.codec.state_dict().items()}
    save_file(sd, str(out.root / "codec_export/nanocodec-22khz-1.89kbps-21.5fps.safetensors"))
    print(f"  [saved] codec_export/…safetensors  ({len(sd)} tensors)")
    out.manifest["artifacts"]["codec_export/state_dict"] = {
        "tensors": len(sd),
        "keys_head": sorted(sd.keys())[:8],
    }
    from omegaconf import OmegaConf
    cfg_obj = getattr(player.codec, "cfg", None) or getattr(player.codec, "_cfg")
    out.text("codec_export/codec_config.yaml", OmegaConf.to_yaml(cfg_obj))
    out.text("codec_export/key_manifest.json", json.dumps(
        {k: list(v.shape) for k, v in sd.items()}, indent=2))

    # -- showcase WAVs (audition; sampled, seeded — NOT parity artifacts) --------
    if not args.skip_showcase:
        print("\n=== showcase (sampled, temp 0.3, seed 0) ===")
        for stem, rc_codes in refs.items():
            for tname in ("canonical", "one_word"):
                torch.manual_seed(0)
                with torch.inference_mode():
                    toks = runner.generate(texts[tname], ref_codes=rc_codes.to(dev),
                                           temperature=0.3)
                    _, w = player.decode(toks)
                out.wav(f"showcase/{stem}_{tname}", w, sr)
            # CFG short-word rescue sample (the on-device policy we plan to ship)
            torch.manual_seed(0)
            with torch.inference_mode():
                toks = runner.generate(texts["one_word"], ref_codes=rc_codes.to(dev),
                                       temperature=0.3, cfg_scale=2.6, cfg_frames=20)
                _, w = player.decode(toks)
            out.wav(f"showcase/{stem}_one_word_cfg", w, sr)

    # -- manifest ---------------------------------------------------------------
    (out.root / "manifest.json").write_text(json.dumps(out.manifest, indent=2))
    print(f"\nmanifest: {len(out.manifest['artifacts'])} artifacts")
    print("ALL CAPTURES OK")
    print(f"\nNext:  cd /workspace && tar czf gepard-goldens.tgz "
          f"{Path(args.out).name} capture.log && sha256sum gepard-goldens.tgz")


if __name__ == "__main__":
    main()
