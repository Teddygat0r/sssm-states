"""
Throwaway probe: introspect a model's KV/recurrent cache so we know exactly
what to deep-copy / repeat / overwrite when building the batched compressed-state
forward in run_kl_generation.py.

Usage:
    python _cache_probe.py mamba2
    python _cache_probe.py qwen35
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RECON = HERE.parent / "reconstruction_experiments"
sys.path.insert(0, str(RECON))

from run_reconstruction import MODELS, DEVICE  # noqa: E402


def tshape(v):
    return tuple(v.shape) if torch.is_tensor(v) else type(v).__name__


def describe_layer(li, layer):
    public = [a for a in dir(layer) if not a.startswith("_")]
    tensors = {a: tshape(getattr(layer, a)) for a in public
               if torch.is_tensor(getattr(layer, a, None))}
    print(f"  layer[{li}] {type(layer).__name__}")
    if tensors:
        print(f"     tensors: {tensors}")
    st = getattr(layer, "state", None)
    if isinstance(st, dict):
        print(f"     state(dict): "
              f"{ {k: tshape(v) for k, v in st.items()} }")
    # non-tensor, non-callable public attrs (e.g. dtype/lengths)
    misc = {a: getattr(layer, a) for a in public
            if not torch.is_tensor(getattr(layer, a, None))
            and not callable(getattr(layer, a, None))
            and a != "state"}
    if misc:
        print(f"     misc: { {k: (tshape(v) if not isinstance(v, (int, float, bool, str, type(None))) else v) for k, v in misc.items()} }")


def main():
    key = sys.argv[1]
    spec = MODELS[key]
    print(f"== loading {spec['label']} ({key}) ==")
    model, tok, state_fn, info = spec["loader"]()
    print(f"   {info}")

    text = "The quick brown fox jumps over the lazy dog. " * 12
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    print(f"   prefill ids: {tuple(ids.shape)}")

    with torch.inference_mode():
        out = model(ids, use_cache=True)

    cache = None
    cache_attr = None
    for attr in ("past_key_values", "cache_params"):
        c = getattr(out, attr, None)
        if c is not None:
            cache, cache_attr = c, attr
            break
    print(f"\n-- cache via out.{cache_attr}: {type(cache).__module__}.{type(cache).__name__}")

    batch_methods = [m for m in dir(cache)
                     if any(t in m.lower() for t in ("batch", "crop", "reorder", "repeat", "select"))
                     and callable(getattr(cache, m, None))]
    print(f"   batch-ish methods: {batch_methods}")

    has_layers = hasattr(cache, "layers")
    print(f"   has .layers: {has_layers}")
    if has_layers:
        print(f"   n layers: {len(cache.layers)}")
        # show the first few layers, and the first recurrent + first attention-ish layer
        shown = 0
        rec_shown = att_shown = False
        for li, layer in enumerate(cache.layers):
            is_rec = (getattr(layer, "recurrent_states", None) is not None) or \
                     (isinstance(getattr(layer, "state", None), dict)
                      and getattr(layer, "state").get("recurrent_state") is not None)
            is_att = (getattr(layer, "keys", None) is not None) or \
                     (getattr(layer, "key_cache", None) is not None)
            if li < 3 or (is_rec and not rec_shown) or (is_att and not att_shown):
                describe_layer(li, layer)
                shown += 1
                rec_shown = rec_shown or is_rec
                att_shown = att_shown or is_att
            if shown >= 6 and rec_shown:
                break
    else:
        # legacy/flat cache: list attrs
        for f in ("key_cache", "value_cache", "conv_states", "recurrent_states",
                  "ssm_states"):
            v = getattr(cache, f, None)
            if v is not None:
                print(f"   cache.{f}: list[{len(v)}] e.g. {tshape(v[0]) if len(v) else 'empty'}")

    # try a single-token decode step to confirm the forward signature works
    print("\n-- single decode step --")
    next_id = out.logits[:, -1:, :].argmax(-1)
    cache_pos = torch.tensor([ids.shape[1]], device=DEVICE)
    try:
        with torch.inference_mode():
            out2 = model(input_ids=next_id, past_key_values=cache,
                         cache_position=cache_pos, use_cache=True)
        print(f"   OK with past_key_values=, cache_position=; logits {tuple(out2.logits.shape)}")
    except Exception as e:
        print(f"   FAILED past_key_values+cache_position: {type(e).__name__}: {e}")
        try:
            with torch.inference_mode():
                out2 = model(input_ids=next_id, **{cache_attr: cache}, use_cache=True)
            print(f"   OK with {cache_attr}=; logits {tuple(out2.logits.shape)}")
        except Exception as e2:
            print(f"   FAILED {cache_attr}=: {type(e2).__name__}: {e2}")


if __name__ == "__main__":
    main()
