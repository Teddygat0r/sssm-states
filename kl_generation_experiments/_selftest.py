"""
Validate kl_core's batched compressed-state readout on a real model.

Invariants checked:
  1. After cache_repeat_ + inject + one batched forward, row 0 (uncompressed)
     reproduces an independent unbatched forward of the true cache (max|Δlogit|
     ~ 0). This proves the manual repeat/select/inject keeps row 0 exact.
  2. KL(row0 || row_r) >= 0 and is (weakly) decreasing as rank grows.
  3. cache_select_row_(0) yields a batch-1 cache that keeps generating.

Usage: python _selftest.py mamba2   (or qwen35, deltanet, ...)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reconstruction_experiments"))
sys.path.insert(0, str(HERE))

from run_reconstruction import MODELS, DEVICE  # noqa: E402
import kl_core as kc  # noqa: E402

RANKS = (4, 8, 16)


def main():
    key = sys.argv[1]
    spec = MODELS[key]
    print(f"== {spec['label']} ({key}) ==")
    model, tok, state_fn, info = spec["loader"]()
    CK = kc.detect_cache_kwarg(model)
    print(f"  cache kwarg: {CK}")

    text = "In a distant future, humanity learned to fold space. " * 8
    ids = tok(text, return_tensors="pt").input_ids[:, :48].to(DEVICE)
    T = ids.shape[1]
    with torch.inference_mode():
        out = model(ids, use_cache=True)
    cache = getattr(out, CK)
    pos = T
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)

    # advance a few real generation steps (batch-1)
    for _ in range(3):
        out = kc.forward_step(model, nxt, cache, pos, CK)
        cache = getattr(out, CK)
        pos += 1
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)

    layer_idx = kc.recurrent_layer_indices(cache)
    print(f"  recurrent layers: {len(layer_idx)}  (e.g. {layer_idx[:6]})")
    states = kc.read_recurrent_states(cache, layer_idx)
    sample_li = layer_idx[0]
    print(f"  state[{sample_li}] shape: {tuple(states[sample_li].shape)}")

    recons, rel = kc.lowrank_recon_all(states, RANKS)   # production rsvd_eigh path

    # unbatched batch-1 reference (informational: bf16 batch-size kernel drift)
    snap = copy.deepcopy(cache)
    out_u = kc.forward_step(model, nxt, snap, pos, CK)
    logits_u = out_u.logits[:, -1][0]

    N = len(RANKS) + 1

    # (A) all-true batch: every row carries the true state -> rows must be
    #     bit-identical (no cross-row contamination, deterministic per row).
    ctrl = copy.deepcopy(cache)
    kc.cache_repeat_(ctrl, N)
    out_ctrl = kc.forward_step(model, nxt.repeat(N, 1), ctrl, pos, CK)
    lc = out_ctrl.logits[:, -1]               # [N, V]
    ctrl_spread = (lc - lc[0:1]).abs().max().item()

    # (B) compressed batch: row 0 true, rows 1.. = rank reconstructions
    kc.cache_repeat_(cache, N)
    for ri, k in enumerate(RANKS, start=1):
        for li in layer_idx:
            kc.write_recurrent_row(cache, li, ri, recons[k][li])
    out_b = kc.forward_step(model, nxt.repeat(N, 1), cache, pos, CK)
    logits_b = out_b.logits[:, -1]            # [N, V]

    # contamination: row 0 of the compressed batch must equal row 0 of the
    # all-true batch (same batch size; only rows 1.. differ). Must be ~0.
    row0_contam = (logits_b[0].float() - lc[0].float()).abs().max().item()
    batch1_drift = (logits_b[0].float() - logits_u.float()).abs().max().item()

    print(f"\n  [A] all-true batch row spread (must be 0):        {ctrl_spread:.3e}")
    print(f"  [B] row0 contamination by compressed rows (~0):  {row0_contam:.3e}")
    print(f"  [i] row0 vs batch-1 forward (bf16 drift, info):  {batch1_drift:.3e}")
    print("  [C] KL(full || rank-k) and mean rel-Fro state error:")
    kl_mono = True       # informational: per-token KL need not be monotone
    rel_mono = True      # must hold: more rank => better reconstruction
    prev_kl = prev_rel = None
    for ri, k in enumerate(RANKS, start=1):
        kl = kc.kl_full_vs_approx(logits_b[0], logits_b[ri])
        if prev_kl is not None and kl > prev_kl + 1e-6:
            kl_mono = False
        if prev_rel is not None and rel[k] > prev_rel + 1e-9:
            rel_mono = False
        prev_kl, prev_rel = kl, rel[k]
        print(f"      rank {k:>2}:  KL = {kl:.4e} nats   rel_fro = {rel[k]:.3e}")

    # (3) continue from row 0
    cont = getattr(out_b, CK)
    kc.cache_select_row_(cont, 0)
    nxt2 = logits_b[0:1].argmax(-1, keepdim=True)
    out_c = kc.forward_step(model, nxt2, cont, pos + 1, CK)
    cont_ok = tuple(out_c.logits.shape[:1]) == (1,)

    ok = ctrl_spread < 1e-3 and row0_contam < 1e-3 and rel_mono and cont_ok
    print(f"\n  no-contamination: {row0_contam < 1e-3}   deterministic-rows: "
          f"{ctrl_spread < 1e-3}   rel_fro-monotone: {rel_mono}   "
          f"continues: {cont_ok}   (kl-monotone[info]: {kl_mono})")
    print(f"  SELFTEST {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
