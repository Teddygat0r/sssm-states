"""Durable five-model suite: validate every model, then run complete evaluations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODELS=['deltanet','gated_deltanet','qwen35','mamba2','nemotron']


def save(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    p=args.output;p.mkdir(parents=True,exist_ok=True)
    status=dict(state='preparing',started_unix=time.time(),completed_jobs=0,total_jobs=10)
    save(p/'status.json',status)
    try:
        os.environ['SWEEP_N_PROMPTS']='100'
        sys.path.insert(0,str(ROOT/'svd_sweep'))
        from _helpers import load_prompts
        prompts=load_prompts()
        assert len(prompts)==200 and all(sum(x['kind']==k for x in prompts)==100 for k in ('chat','code'))
        save(p/'prompts.json',prompts)
        save(p/'config.json',dict(models=MODELS,methods=['int8_layer_affine']+[f'svd{r}' for r in [1,2,4,8,16,32]],ranks=[1,2,4,8,16,32],prompts_per_kind=100,
            positions=[256,'full'],gen_tokens=128,every=4,expected_measurements_per_model_method_position=6400,
            paired=True,estimated_hours=[6,14],
            code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'worker.py',HERE/'suite.py')}))
        for phase in ('smoke','full'):
            for model in MODELS:
                dest=p/phase/model;dest.mkdir(parents=True,exist_ok=True)
                status.update(state='running',phase=phase,model=model,current_job=str(dest),updated_unix=time.time());save(p/'status.json',status)
                cmd=[sys.executable,str(HERE/'worker.py'),'--model',model,'--prompts',str(p/'prompts.json'),'--output',str(dest)]
                if phase=='smoke':cmd.append('--smoke')
                print(f'START {phase} {model}',flush=True)
                with (dest/'run.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                s=json.loads((dest/'status.json').read_text());assert s['state']=='completed'
                status['completed_jobs']+=1;save(p/'status.json',status)
                print(f'DONE {phase} {model}',flush=True)
                if phase=='full':
                    merged=[]
                    for f in sorted((p/'full').glob('*/summary.json')):merged.extend(json.loads(f.read_text()))
                    save(p/'summary.json',merged)
                    with (p/'summary.csv').open('w') as f:
                        w=csv.DictWriter(f,fieldnames=list(merged[0]));w.writeheader();w.writerows(merged)
        with (p/'comparison.md').open('w') as f:
            f.write('# Single-step KL comparison\n\n| Model | Prefix | Method | N | Mean KL | Median | p90 | p99 | Top-1 agreement |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n')
            for r in merged:
                f.write(f"| {r['model']} | {'full' if r['position']==-1 else r['position']} | {r['method']} | {r['n']} | {r['kl_mean']:.6g} | {r['kl_median']:.6g} | {r['kl_p90']:.6g} | {r['kl_p99']:.6g} | {r['top1_agreement']:.3%} |\n")
        status.update(state='completed',finished_unix=time.time());save(p/'status.json',status)
        print('SUITE COMPLETED',flush=True)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished_unix=time.time());save(p/'status.json',status);raise


if __name__=='__main__':main()
