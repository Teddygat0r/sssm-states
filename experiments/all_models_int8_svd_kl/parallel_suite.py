"""Run five independent model processes concurrently on the shared GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from suite import HERE,MODELS,save


def run_worker(model,phase,p,per_head=False):
    dest=p/phase/model;dest.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(HERE/'worker.py'),'--model',model,'--prompts',str(p/'prompts.json'),'--output',str(dest)]
    if phase=='smoke':cmd.append('--smoke')
    if per_head:cmd.append('--int8-per-head-only')
    print(f'START {phase} {model}',flush=True)
    with (dest/'run.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        save(dest/'process.json',dict(pid=proc.pid,started_unix=time.time(),command=cmd))
        code=proc.wait()
    if code:raise RuntimeError(f'{phase}/{model} exited {code}: {dest}/run.log')
    state=json.loads((dest/'status.json').read_text())
    if state['state']!='completed':raise RuntimeError(f'{model} did not complete')
    print(f'DONE {phase} {model}',flush=True)
    return model


def merge(p):
    rows=[]
    for f in sorted((p/'full').glob('*/summary.json')):rows.extend(json.loads(f.read_text()))
    if not rows:return
    save(p/'summary.json',rows)
    with (p/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (p/'comparison.md').open('w') as f:
        f.write('# Single-step compression KL\n\n| Model | Prefix | Method | N | Mean KL | p99 KL | Top-1 agreement |\n|---|---|---|---:|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['model']} | {'full' if r['position']==-1 else r['position']} | {r['method']} | {r['n']} | {r['kl_mean']:.6g} | {r['kl_p99']:.6g} | {r['top1_agreement']:.3%} |\n")

    with (p/'RESULTS.md').open('w') as f:
        f.write('# Single-step compression results\n\nSame 100 chat and 100 code prompts, prefixes 256/full, 128 greedy tokens, probes every four steps. Probes are discarded; attention and convolution states are unchanged. Per-head INT8 uses one affine scale and zero-point per matrix, per head, per layer. No SVD is run in per-head-only mode. Batch size is two in that mode; numerical differences from the previous eight-row probe batch may affect reference trajectories.\n\n')
        f.write('| Model | Prefix | Method | N | KL mean | Median | p90 | p99 | Max | Top-1 agreement | Mean squared relative Frobenius error |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for r in rows:
            f.write('| '+ ' | '.join(str(r[k]) if k in ('model','position','method','n') else f'{r[k]:.8g}' for k in ('model','position','method','n','kl_mean','kl_median','kl_p90','kl_p99','kl_max','top1_agreement','state_rel_fro_squared_mean'))+' |\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prompts',type=Path,required=True)
    ap.add_argument('--int8-per-head-only',action='store_true');args=ap.parse_args()
    p=args.output;p.mkdir(parents=True,exist_ok=True)
    status=dict(state='starting',started_unix=time.time(),completed_jobs=0,total_jobs=10,
                concurrency=5,completed_models=[],failed_models={})
    save(p/'status.json',status)
    try:
        prompts=json.loads(args.prompts.read_text())
        assert len(prompts)==200 and all(sum(x['kind']==k for x in prompts)==100 for k in ('chat','code'))
        save(p/'prompts.json',prompts)
        save(p/'config.json',dict(models=MODELS,concurrency=5,separate_processes=True,
            methods=(['int8_head_affine'] if args.int8_per_head_only else ['int8_layer_affine']+[f'svd{r}' for r in [1,2,4,8,16,32]]),ranks=([] if args.int8_per_head_only else [1,2,4,8,16,32]),prompts_per_kind=100,gen_tokens=128,every=4,positions=[256,'full'],
            expected_measurements_per_model_method_position=6400,source_prompts=str(args.prompts),
            source_prompt_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            estimated_hours=([3,6] if args.int8_per_head_only else [6,14]),prefill_logits='last token only when supported',
            code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'worker.py',Path(__file__))}))
        for phase in ('smoke','full'):
            status.update(state='running',phase=phase,active_models=list(MODELS),completed_models=[],failed_models={})
            save(p/'status.json',status)
            # Threads only supervise subprocess lifecycles; each model has its own Python/CUDA process.
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures={pool.submit(run_worker,m,phase,p,args.int8_per_head_only):m for m in MODELS}
                for future in as_completed(futures):
                    model=futures[future]
                    try:
                        future.result();status['completed_jobs']+=1;status['completed_models'].append(model)
                    except Exception:
                        status['failed_models'][model]=traceback.format_exc()
                    status['active_models'].remove(model)
                    status['updated_unix']=time.time();save(p/'status.json',status)
                    if phase=='full':merge(p)
            if status['failed_models']:raise RuntimeError(f"Failed {phase} models: {list(status['failed_models'])}")
        status.update(state='completed',finished_unix=time.time());save(p/'status.json',status)
        print('PARALLEL SUITE COMPLETED',flush=True)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished_unix=time.time());save(p/'status.json',status);raise


if __name__=='__main__':main()
