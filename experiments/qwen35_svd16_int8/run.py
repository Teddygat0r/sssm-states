"""Repeated SSM factor compression: reference likelihood + free continuations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'kl_generation_experiments'),
                str(ROOT / 'reconstruction_experiments'), str(ROOT / 'svd_sweep')]
import torch
import torch.nn.functional as F
import kl_core as kc
from rsvd_eigh import randomized_svd_eigh

ARMS = ['uncompressed', 'svd16', 'svd16_int8']


def quantize(x, dim):
    # One symmetric scale per head and rank component, FP32 scale metadata.
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-30) / 127
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q.float() * scale


def compress(cache, indices, seed):
    # Rows 0..2 teacher-force references; rows 3..5 freely generate.
    rows = [1, 2, 4, 5]
    states = torch.stack([cache.layers[i].recurrent_states[rows].float() for i in indices])
    # Reset sketch seed per row so identical inputs get identical sketches.
    recon = []
    for j in range(4):
        torch.manual_seed(seed)
        u, s, vh = randomized_svd_eigh(states[:, j], rank=16, oversample=8, n_iter=2)
        left = u * s.sqrt().unsqueeze(-2)
        right = s.sqrt().unsqueeze(-1) * vh
        if j in (1, 3):
            left, right = quantize(left, -2), quantize(right, -1)
        recon.append(left @ right)
    restored = torch.stack(recon, dim=1)
    if not torch.isfinite(restored).all():
        raise RuntimeError('Nonfinite compressed state')
    for j, li in enumerate(indices):
        cache.layers[li].recurrent_states[rows] = restored[j].to(cache.layers[li].recurrent_states.dtype)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


@torch.inference_mode()
def run_task(model, tok, ids, context, steps, seed, check=False):
    started = time.monotonic()
    out = model(input_ids=ids[:context-1].unsqueeze(0).cuda(), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    indices = kc.recurrent_layer_indices(cache)
    assert indices, 'No recurrent layers found'
    kc.cache_repeat_(cache, 6)
    token = ids[context-1].reshape(1, 1).cuda().repeat(6, 1)
    generated = [[], [], []]
    ended = [False] * 3
    eos = tok.eos_token_id
    nll, kl, agreement = [], [], []
    for step in range(steps):
        if check and step == 0:
            import copy
            probe = copy.deepcopy(cache)
            logits = kc.forward_step(model, token, probe, context-1, 'past_key_values').logits[:, -1].float()
            assert torch.equal(logits[0], logits[1]) and torch.equal(logits[0], logits[5]), 'Baseline batch mismatch'
            baseline_logits = logits[0].clone()
            before = [cache.layers[i].recurrent_states[[0, 3]].clone() for i in indices]
        compress(cache, indices, seed + step)
        if check and step == 0:
            assert all(torch.equal(cache.layers[i].recurrent_states[[0, 3]], b) for i, b in zip(indices, before))
        out = kc.forward_step(model, token, cache, context-1+step, 'past_key_values')
        cache = out.past_key_values
        logits = out.logits[:, -1].float()
        assert torch.isfinite(logits).all(), 'Nonfinite logits'
        if check and step == 0:
            assert torch.equal(logits[0], logits[3]), 'Baseline row contamination'
            assert torch.equal(logits[0], baseline_logits), 'Compression changed baseline logits'
            print('CHECKS PASSED: equal baseline rows, untouched control states, finite compressed states/logits', flush=True)
        target = ids[context+step].cuda()
        lp = logits[:3].log_softmax(-1)
        nll.append((-lp[:, target]).cpu())
        kl.append(torch.stack([F.kl_div(lp[j], lp[0], reduction='sum', log_target=True) for j in (1, 2)]).cpu())
        agreement.append((logits[1:3].argmax(-1) == logits[0].argmax(-1)).float().cpu())
        nxt = logits[3:].argmax(-1)
        for j, t in enumerate(nxt.tolist()):
            if not ended[j]:
                generated[j].append(t)
                ended[j] = t == eos
        token = torch.cat([target.repeat(3), nxt]).reshape(6, 1)
    mean_nll = torch.stack(nll).mean(0).tolist()
    mean_kl = torch.stack(kl).mean(0).tolist()
    agree = torch.stack(agreement).mean(0).tolist()
    return dict(seconds=time.monotonic()-started, recurrent_layers=len(indices), context_tokens=context,
                scored_tokens=steps, prompt_token_ids=ids[:context].tolist(),
                prompt=tok.decode(ids[:context]), reference=tok.decode(ids[context:context+steps]),
                reference_token_ids=ids[context:context+steps].tolist(),
                arms={name: dict(nll=mean_nll[j], perplexity=math.exp(mean_nll[j]),
                                 kl_from_uncompressed=0 if j == 0 else mean_kl[j-1],
                                 top1_agreement=1 if j == 0 else agree[j-1],
                                 generated_token_ids=generated[j],
                                 generated_text=tok.decode(generated[j], skip_special_tokens=True),
                                 stopped='eos' if ended[j] else 'max_tokens') for j, name in enumerate(ARMS)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--prompts-per-kind', type=int, default=10)
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--contexts', type=int, nargs='+', default=[256, 2048])
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(state='starting', started_unix=time.time(), completed=0)
    write_json(args.output / 'status.json', status)
    try:
        os.environ['SWEEP_N_PROMPTS'] = str(args.prompts_per_kind)
        from _helpers import load_prompts
        from run_reconstruction import load_qwen35
        torch.set_num_threads(8)
        torch.manual_seed(20260921)
        prompts = load_prompts()
        assert {'chat', 'code'} == {p['kind'] for p in prompts}, 'Both prompt sources required'
        if not args.smoke:
            assert all(sum(p['kind'] == kind for p in prompts) == args.prompts_per_kind for kind in ('chat', 'code'))
        write_json(args.output / 'prompts.json', prompts)
        print('Loading Qwen3.5-4B', flush=True)
        model, tok, _, info = load_qwen35()
        print(info, flush=True)
        tasks = []
        for p in prompts:
            ids = tok(p['text'], return_tensors='pt').input_ids[0]
            for context in args.contexts:
                assert len(ids) >= context + args.steps
                tasks.append((p, ids, context))
        if args.smoke:
            tasks = tasks[:1]
        cfg = dict(model='Qwen/Qwen3.5-4B', arms=ARMS, rank=16, bits=8, seed=20260921,
                   factorization='L=U*sqrt(S), R=sqrt(S)*Vh; independently symmetric int8 per rank component',
                   compression='after prefill and every decode step; all recurrent layers; reconstruct for dense compute',
                   metrics='teacher-forced reference NLL/perplexity, KL and top1 vs uncompressed; independent greedy text',
                   attention_and_conv='uncompressed', contexts=args.contexts, steps=args.steps,
                   prompts_per_kind=args.prompts_per_kind, tasks=len(tasks), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(), smoke=args.smoke,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        write_json(args.output / 'config.json', cfg)
        records = []
        for i, (p, ids, context) in enumerate(tasks):
            status.update(state='running', total=len(tasks), current_prompt=p['id'], current_context=context)
            write_json(args.output / 'status.json', status)
            record = run_task(model, tok, ids, context, args.steps, 20260921+i*1000, check=i == 0)
            record.update(prompt_id=p['id'], kind=p['kind'])
            with (args.output / 'samples.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            records.append(record)
            avg = sum(r['seconds'] for r in records) / len(records)
            status.update(completed=i+1, mean_task_seconds=avg,
                          eta_unix=time.time()+avg*(len(tasks)-i-1), updated_unix=time.time())
            write_json(args.output / 'status.json', status)
            print(f"[{i+1}/{len(tasks)}] {p['id']} context={context}: {record['seconds']:.1f}s; remaining ~{avg*(len(tasks)-i-1)/60:.1f} min", flush=True)
        summary = {}
        for kind in ['all', 'chat', 'code']:
            subset = [r for r in records if kind == 'all' or r['kind'] == kind]
            if not subset:
                continue
            summary[kind] = {}
            for arm in ARMS:
                metrics = {k: sum(r['arms'][arm][k] for r in subset)/len(subset)
                           for k in ('nll', 'kl_from_uncompressed', 'top1_agreement')}
                metrics['perplexity'] = math.exp(metrics['nll'])
                summary[kind][arm] = metrics
        write_json(args.output / 'summary.json', summary)
        with (args.output / 'outputs.md').open('w') as f:
            for r in records:
                f.write(f"## {r['prompt_id']} / {r['context_tokens']} tokens\n\n")
                for arm in ARMS:
                    f.write(f"### {arm}\n\n{r['arms'][arm]['generated_text']}\n\n")
        status.update(state='completed', finished_unix=time.time())
        write_json(args.output / 'status.json', status)
        print('COMPLETED', flush=True)
    except BaseException:
        status.update(state='failed', error=traceback.format_exc(), finished_unix=time.time())
        write_json(args.output / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
