"""Explain why Hadamard-rotated 4-bit quant collapses GSM8K accuracy.

`hadamard_q4` scored 0.363 strict-match vs naive `q4`'s 0.807 (see
gsm8k_results.csv), yet the rotation math is correct and reconstructs the state
*better* than naive quant. This script reproduces that paradox on a REAL
Qwen3.5-4B recurrent state and shows the resolution: the failure is a
decode-dynamics problem, not a reconstruction problem (see README.md).

It captures the recurrent state after prefilling a GSM8K-style prompt, then
compares naive per-head q4 vs Hadamard-rotated q4 along three axes:
  - distribution (dynamic range / kurtosis) of the values fed to the quantizer
  - reconstruction error, globally (Frobenius) and on the largest-|entries|
  - reconstruction error INSIDE the rank-r dominant singular subspace that
    linear attention actually reads via o = q @ S

The self-test in compression_ops.py uses Gaussian noise, which is invariant
under Hadamard rotation and therefore cannot reproduce the failure; a real,
outlier-dominated state is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compression_ops import (  # noqa: E402
    fake_quant_per_head,
    hadamard_matrix,
    hadamard_quant,
)

MODEL_ID = "Qwen/Qwen3.5-4B"
DEVICE = "cuda"

PROMPT = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she "
    "sold half as many clips in May. How many clips did she sell altogether in "
    "April and May?\nAnswer: Natalia sold 48/2 = 24 clips in May. Altogether she "
    "sold 48+24 = 72 clips. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
    "minutes of babysitting. How much did she earn?\nAnswer:"
)


def stats(name: str, x: torch.Tensor) -> None:
    x = x.float().flatten()
    mean = x.mean()
    std = x.std()
    absmax = x.abs().max()
    # excess kurtosis
    z = (x - mean) / (std + 1e-12)
    kurt = (z.pow(4).mean() - 3.0)
    # outlier ratio: how big is the largest value relative to the std
    print(
        f"  {name:28s} std={std.item():.4g}  absmax={absmax.item():.4g}  "
        f"absmax/std={ (absmax/(std+1e-12)).item():7.1f}  "
        f"excess_kurt={kurt.item():9.1f}"
    )


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE)
    model.eval()

    ids = tok.encode(PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, return_dict=True)
    cache = out.past_key_values

    states = []
    for layer in cache.layers:
        t = getattr(layer, "recurrent_states", None)
        if torch.is_tensor(t) and t.dim() >= 4:
            states.append(t)
    S = torch.cat(states, dim=0).float()   # [L_lin*B, H, D_k, D_v]
    print(f"\nCaptured recurrent state: {tuple(S.shape)}  dtype was bf16")

    Hk = hadamard_matrix(S.shape[-2], S.device)
    Hv = hadamard_matrix(S.shape[-1], S.device)
    R = Hk @ S @ Hv

    print("\nDistribution of values fed to the quantizer (flattened over all):")
    stats("original state", S)
    stats("Hadamard-rotated state", R)

    print("\nPer-head dynamic range (drives the per-head quant scale):")
    # For each head, range = max-min over its 128x128 block.
    def head_range(x):
        flat = x.flatten(start_dim=-2)
        return (flat.amax(-1) - flat.amin(-1))
    rng_orig = head_range(S)
    rng_rot = head_range(R)
    # The damage metric: how much does the per-head range grow under rotation,
    # while the "typical" magnitude (std) that we actually care about resolving
    # stays similar? Bigger range / same signal = coarser effective step.
    print(f"  mean per-head range  original={rng_orig.mean().item():.4g}  "
          f"rotated={rng_rot.mean().item():.4g}  "
          f"ratio={ (rng_rot.mean()/rng_orig.mean()).item():.2f}x")

    print("\nReconstruction error (rel-Frobenius over full state):")
    for bits in (8, 4):
        q = fake_quant_per_head(S, bits)
        hq = hadamard_quant(S, bits)
        eq = (q - S).norm() / S.norm()
        ehq = (hq - S).norm() / S.norm()
        print(f"  {bits}-bit:  naive_per_head={eq.item():.4f}   "
              f"hadamard={ehq.item():.4f}   "
              f"(hadamard is {ehq.item()/eq.item():.2f}x the naive error)")

    # The decisive metric: error INSIDE the signal subspace. Linear attention
    # reads o = q @ S, and the SVD sweep showed S is ~rank-4 (r=4 == baseline),
    # so only the top few singular directions carry information. Frobenius error
    # counts the (huge, irrelevant) near-null space; what matters is how each
    # method corrupts the dominant subspace.
    print("\nError restricted to the dominant singular subspace (per head, "
          "averaged):")
    Sp = S.reshape(-1, S.shape[-2], S.shape[-1])     # [N_head, D_k, D_v]
    q = fake_quant_per_head(S, 4).reshape_as(Sp)
    hq = hadamard_quant(S, 4).reshape_as(Sp)
    # Per-head top-r left singular vectors (row/query space that q @ S reads).
    U, _, _ = torch.linalg.svd(Sp, full_matrices=False)
    for r in (1, 4, 16):
        Ur = U[:, :, :r]                              # [N, D_k, r]
        def proj_err(hat):
            e = hat - Sp
            # project error & signal onto top-r left subspace: Ur^T @ M
            en = (Ur.transpose(-1, -2) @ e).norm(dim=(-2, -1))
            sn = (Ur.transpose(-1, -2) @ Sp).norm(dim=(-2, -1))
            return (en / (sn + 1e-12)).mean().item()
        print(f"  rank-{r:<2d} subspace rel-err:  naive={proj_err(q):.4f}  "
              f"hadamard={proj_err(hq):.4f}  "
              f"({proj_err(hq)/max(proj_err(q),1e-9):.2f}x)")

    # The key question: WHERE does each method put its error? The state is
    # outlier-dominated, and those few huge entries likely carry the signal.
    # Naive per-head quant sets its range by the outlier, so the outlier itself
    # lands in the top bin and is preserved; Hadamard smears each outlier across
    # all 128 coords, so quant noise rotates back ONTO the outlier.
    Sf = S.flatten()
    absS = Sf.abs()
    for bits in (4,):
        q = fake_quant_per_head(S, bits).flatten()
        hq = hadamard_quant(S, bits).flatten()
        for pct, label in [(0.001, "top 0.1%"), (0.01, "top 1%"), (1.0, "all")]:
            k = max(1, int(absS.numel() * pct))
            idx = torch.topk(absS, k).indices
            sig = Sf[idx]
            eq = (q[idx] - sig).norm() / sig.norm()
            ehq = (hq[idx] - sig).norm() / sig.norm()
            print(f"  {bits}-bit rel-err on {label:9s} largest-|entries| "
                  f"({k:>7d} vals): naive={eq.item():.4f}  "
                  f"hadamard={ehq.item():.4f}  "
                  f"({ehq.item()/max(eq.item(),1e-9):.2f}x)")


if __name__ == "__main__":
    main()
