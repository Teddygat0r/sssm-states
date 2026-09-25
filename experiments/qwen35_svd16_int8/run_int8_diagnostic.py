"""Paired GSM8K pilot separating quantization granularity and frequency."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback
import torch
import run_gsm8k as base
from run import quantize, write_json, randomized_svd_eigh

ARMS = ['uncompressed', 'int8_head_every', 'int8_row_every',
        'int8_column_every', 'int8_head_prefill_only', 'svd16_int8_every']


def reconstruct(states, arm, seed):
    if arm in ('int8_head_every', 'int8_head_prefill_only'):
        return quantize(states.flatten(-2), -1).reshape_as(states)
    if arm == 'int8_row_every':
        return quantize(states, -1)
    if arm == 'int8_column_every':
        return quantize(states, -2)
    if arm == 'svd16_int8_every':
        torch.manual_seed(seed)
        u, s, vh = randomized_svd_eigh(states, rank=16, oversample=8, n_iter=2)
        return quantize(u*s.sqrt().unsqueeze(-2), -2) @ quantize(s.sqrt().unsqueeze(-1)*vh, -1)
    raise ValueError(arm)


class DiagnosticCompression:
    def __init__(self):
        self.step = 0
        self.metrics = {}

    def __call__(self, cache, indices, seed, ended):
        for row, arm in enumerate(ARMS[1:], 1):
            if ended[row] or (arm == 'int8_head_prefill_only' and self.step > 0):
                continue
            states = torch.stack([cache.layers[i].recurrent_states[row].float() for i in indices])
            restored = reconstruct(states, arm, seed)
            assert torch.isfinite(restored).all()
            if self.step == 0:
                den = states.square().sum((-2,-1)).clamp_min(1e-30)
                self.metrics[arm] = dict(
                    mean_head_relative_squared_error=((states-restored).square().sum((-2,-1))/den).mean().item(),
                    fraction_nonzero_entries_erased=((states != 0) & (restored == 0)).sum().item()/max(1,(states != 0).sum().item()),
                    maxabs_over_rms_mean=(states.abs().amax((-2,-1))/states.square().mean((-2,-1)).sqrt().clamp_min(1e-30)).mean().item())
            for j, li in enumerate(indices):
                base.kc.write_recurrent_row(cache, li, row, restored[j])
        self.step += 1


def summarize(records):
    n = len(records)
    out = {}
    for arm in ARMS:
        delta = [int(r['arms'][arm]['strict_correct'])-int(r['arms']['uncompressed']['strict_correct']) for r in records]
        mean = sum(delta)/n
        se = math.sqrt(sum((d-mean)**2 for d in delta)/(n-1)/n) if n > 1 else None
        out[arm] = dict(n=n, strict_accuracy=sum(r['arms'][arm]['strict_correct'] for r in records)/n,
                       flexible_accuracy=sum(r['arms'][arm]['flexible_correct'] for r in records)/n,
                       truncated=sum(r['arms'][arm]['stopped']=='max_tokens' for r in records),
                       mean_generated_tokens=sum(len(r['arms'][arm]['token_ids']) for r in records)/n,
                       paired_accuracy_delta=mean,
                       paired_delta_approx_95ci=[mean-1.96*se, mean+1.96*se] if se is not None else None,
                       baseline_correct_arm_wrong=sum(d == -1 for d in delta),
                       baseline_wrong_arm_correct=sum(d == 1 for d in delta))
        if arm != 'uncompressed':
            out[arm]['prefill_state_metrics'] = {k:sum(r['prefill_state_metrics'][arm][k] for r in records)/n
                                                for k in records[0]['prefill_state_metrics'][arm]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    p = args.output
    p.mkdir(parents=True, exist_ok=True)
    status = dict(state='starting', started_unix=time.time(), completed=0)
    write_json(p/'status.json',status)
    try:
        torch.set_num_threads(8)
        questions = base.load_questions()
        selected = sorted(random.Random(20260922).sample(range(len(questions)), args.limit))
        questions = [questions[i] for i in selected]
        if args.smoke:
            questions = questions[:1]
        base.ARMS = ARMS
        write_json(p/'questions.json',questions)
        write_json(p/'config.json',dict(model='Qwen/Qwen3.5-4B',arms=ARMS, sample_seed=20260922,
                   n_questions=len(questions), doc_ids=[q['doc_id'] for q in questions],
                   max_tokens=768, num_fewshot=5, greedy=True, smoke=args.smoke,
                   quantization='symmetric signed INT8 [-127,127]; FP32 scales; dense reconstruction',
                   scales_per_head={'int8_head_every':1,'int8_row_every':128,'int8_column_every':128,
                                    'int8_head_prefill_only':1,'svd16_int8_every':32},
                   compression='before last prompt token and every decode step except prefill_only arm',
                   state_dimensions=[128,128], svd_rank=16, oversample=8, n_iter=2,
                   source=str(base.SOURCE),source_sha256=hashlib.sha256(base.SOURCE.read_bytes()).hexdigest(),
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   base_runner_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()))
        from run_reconstruction import load_qwen35
        print(f'Loading model; {len(questions)} questions, {len(ARMS)} arms',flush=True)
        model,tok,_,_=load_qwen35()
        records=[]
        for q in questions:
            status.update(state='running',total=len(questions),current_doc_id=q['doc_id'],current_decode_steps=0)
            write_json(p/'status.json',status)
            compression=DiagnosticCompression()
            base.compress_three=compression
            def heartbeat(steps,seconds):
                status.update(current_decode_steps=steps,current_seconds=seconds,updated_unix=time.time())
                write_json(p/'status.json',status)
                if not records and steps==4:
                    print(f'STARTUP VERIFIED: {steps} steps in {seconds:.2f}s',flush=True)
            r=base.generate(model,tok,q,768,not records,heartbeat)
            r['prefill_state_metrics']=compression.metrics
            with (p/'samples.jsonl').open('a') as f:
                f.write(json.dumps(r)+'\n')
            records.append(r)
            write_json(p/'summary.json',summarize(records))
            avg=sum(x['seconds'] for x in records)/len(records)
            status.update(completed=len(records),mean_question_seconds=avg,
                          eta_unix=time.time()+avg*(len(questions)-len(records)),updated_unix=time.time())
            write_json(p/'status.json',status)
            print(f"[{len(records)}/{len(questions)}] {r['seconds']:.1f}s, {r['decode_steps']} steps; remaining {avg*(len(questions)-len(records))/3600:.2f}h",flush=True)
        with (p/'outputs.md').open('w') as f:
            for r in records:
                f.write(f"## Question {r['doc_id']}\n\n{r['question']}\n\n")
                for arm in ARMS:
                    f.write(f"### {arm} (correct={r['arms'][arm]['strict_correct']})\n\n{r['arms'][arm]['text']}\n\n")
        status.update(state='completed',finished_unix=time.time())
        write_json(p/'status.json',status)
        print('COMPLETED',flush=True)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished_unix=time.time())
        write_json(p/'status.json',status)
        raise


if __name__=='__main__':
    main()
