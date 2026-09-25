"""Paired single-step INT8/SVD KL probes on an uncompressed reference rollout."""
import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'kl_generation_experiments'),str(ROOT/'reconstruction_experiments'),str(ROOT/'svd_sweep')]
import torch
import kl_core as kc
from run_reconstruction import MODELS,DEVICE
from _helpers import build_tasks,free_memory
RANKS=[1,2,4,8,16,32]
# Match the two historical sweeps' sketch widths (rank_max + oversample).
SVD_GROUPS=[([4,8,16],8),([1,2,32],0)]
METHODS=['int8_layer_affine']+[f'svd{rank}' for rank in RANKS]
PROBE_BATCH=1+len(METHODS)


def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def int8_reconstruct(x):
    x=x.float()
    if METHODS == ['int8_head_affine']:
        assert x.ndim == 3, x.shape  # [heads, rows, columns]
        lo=x.amin(dim=(-2,-1),keepdim=True)
        hi=x.amax(dim=(-2,-1),keepdim=True)
    else:
        lo=x.min();hi=x.max()
    scale=((hi-lo)/255).clamp_min(1e-12)
    zero=(-lo/scale).round().clamp(0,255)
    q=(x/scale+zero).round().clamp(0,255).to(torch.uint8)
    return (q.float()-zero)*scale


def summaries(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['model'],r['position'],r['method'])].append(r)
    result=[]
    for (model,pos,method),rs in sorted(groups.items()):
        vals=sorted(r['kl'] for r in rs)
        pct=lambda q:vals[math.ceil(q*len(vals))-1]
        result.append(dict(model=model,position=pos,method=method,n=len(vals),
            kl_mean=sum(vals)/len(vals),kl_median=pct(.5),kl_p90=pct(.9),kl_p99=pct(.99),kl_max=max(vals),
            top1_agreement=sum(r['top1_agreement'] for r in rs)/len(rs),
            state_rel_fro_squared_mean=sum(r['state_rel_fro_squared'] for r in rs)/len(rs)))
    return result


@torch.inference_mode()
def run_task(model,tok,ck,task,steps,check,seed):
    torch.manual_seed(seed)
    ids=task.ids.unsqueeze(0).to(DEVICE)
    prefill_kwargs={'logits_to_keep':1} if 'logits_to_keep' in inspect.signature(model.forward).parameters else {}
    out=model(input_ids=ids,use_cache=True,**prefill_kwargs)
    cache=getattr(out,ck);nxt=out.logits[:,-1].argmax(-1,keepdim=True)
    indices=kc.recurrent_layer_indices(cache)
    assert indices,'No recurrent states found'
    reference=[];rows=[];checks=None
    for step in range(steps):
        reference.append(int(nxt.item()))
        pos=task.actual_tokens+step
        if step%4==0:
            states=kc.read_recurrent_states(cache,indices)
            assert all(torch.isfinite(x).all() for x in states.values())
            quant={li:int8_reconstruct(x) for li,x in states.items()}
            svd={};rel={}
            for ranks,oversample in SVD_GROUPS:
                group_recon,group_rel=kc.lowrank_recon_all(states,ranks,oversample=oversample,niter=2)
                svd.update(group_recon);rel.update(group_rel)
            recons=[quant]+[svd[rank] for rank in RANKS]
            quant_error=sum(kc.rel_fro_mse(states[li],quant[li])*states[li].shape[0] for li in indices)/sum(states[li].shape[0] for li in indices)
            errors=[quant_error]+[rel[rank] for rank in RANKS]
            assert all(torch.isfinite(x).all() for recon in recons for x in recon.values())
            if check and step==0:
                ctrl=copy.deepcopy(cache);kc.cache_repeat_(ctrl,PROBE_BATCH)
                co=kc.forward_step(model,nxt.repeat(PROBE_BATCH,1),ctrl,pos,ck)
                control_logits=co.logits[:,-1].clone()
                spread=(control_logits-control_logits[:1]).abs().max().item()
                assert spread<1e-3,('unequal controls',spread)
                del co,ctrl
            kc.cache_repeat_(cache,PROBE_BATCH)
            for row,recon in enumerate(recons,1):
                for li in indices:kc.write_recurrent_row(cache,li,row,recon[li])
            out=kc.forward_step(model,nxt.repeat(PROBE_BATCH,1),cache,pos,ck)
            logits=out.logits[:,-1]
            assert torch.isfinite(logits).all()
            if check and step==0:
                drift=(logits[0].float()-control_logits[0].float()).abs().max().item()
                assert drift<1e-3,('reference contamination',drift)
                checks=dict(control_row_spread=spread,reference_contamination=drift,recurrent_layers=len(indices))
                print('CHECKS PASSED '+json.dumps(checks),flush=True)
            top=logits.argmax(-1).tolist()
            for row,method in enumerate(METHODS,1):
                kl=kc.kl_full_vs_approx(logits[0],logits[row])
                assert math.isfinite(kl) and kl>=-1e-5,('invalid KL',kl)
                prob=logits[row].float().softmax(-1)
                vals,idx=prob.topk(5)
                rows.append(dict(model=task.prompt['_model'],prompt_id=task.prompt['id'],input_kind=task.prompt['kind'],
                    position=task.position,prefill_tokens=task.actual_tokens,gen_step=step,predicted_generation_index=step+1,
                    method=method,rank=None if row==1 else RANKS[row-2],kl=kl,state_rel_fro_squared=errors[row-1],top1_agreement=top[0]==top[row],
                    reference_top1_id=top[0],probe_top1_id=top[row],
                    reference_top1_text=tok.decode([top[0]]),probe_top1_text=tok.decode([top[row]]),
                    probe_top5_ids=idx.tolist(),probe_top5_probabilities=vals.tolist()))
            cache=getattr(out,ck);kc.cache_select_row_(cache,0)
            nxt=logits[:1].argmax(-1,keepdim=True)
        else:
            out=kc.forward_step(model,nxt,cache,pos,ck)
            assert torch.isfinite(out.logits[:,-1]).all()
            cache=getattr(out,ck);nxt=out.logits[:,-1].argmax(-1,keepdim=True)
    return rows,dict(prompt_id=task.prompt['id'],input_kind=task.prompt['kind'],position=task.position,
        prefill_tokens=task.actual_tokens,prompt_token_ids=task.ids.tolist(),reference_token_ids=reference,
        reference_text=tok.decode(reference,skip_special_tokens=False),checks=checks)


def main():
    global RANKS,SVD_GROUPS,METHODS,PROBE_BATCH
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True)
    ap.add_argument('--prompts',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--int8-per-head-only',action='store_true');args=ap.parse_args()
    if args.int8_per_head_only:
        RANKS=[];SVD_GROUPS=[];METHODS=['int8_head_affine'];PROBE_BATCH=2
    p=args.output;p.mkdir(parents=True,exist_ok=True)
    status=dict(state='starting',started_unix=time.time(),completed=0,model=args.model)
    save(p/'status.json',status)
    try:
        torch.set_num_threads(1)
        prompts=json.loads(args.prompts.read_text())
        if args.smoke:prompts=[next(x for x in prompts if x['kind']==k) for k in ('chat','code')]
        for prompt in prompts:prompt['_model']=MODELS[args.model]['label']
        steps=8 if args.smoke else 128
        save(p/'config.json',dict(model=MODELS[args.model]['label'],methods=METHODS,prompts=len(prompts),
            gen_tokens=steps,every=4,positions=[256,'full'],max_full_tokens=65536,stop_on_eos=False,
            int8=('FP32 affine min/max with one scale/zero-point per head per layer; 256 levels' if args.int8_per_head_only else 'FP32 affine min/max with one scale/zero-point per layer across all heads; 256 levels'),
            svd=('disabled' if args.int8_per_head_only else 'randomized_svd_eigh, historical rank groups, niter2; no factor quantization'),
            ranks=RANKS,svd_groups=SVD_GROUPS,probe_batch=PROBE_BATCH,
            protocol='paired isolated single-step probes; keep native reference row; attention and conv untouched',
            prefill_logits='last token only when model explicitly supports logits_to_keep',
            prompt_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),smoke=args.smoke))
        print('Loading '+args.model,flush=True)
        model,tok,_,info=MODELS[args.model]['loader']();ck=kc.detect_cache_kwarg(model)
        print(info+' cache='+ck,flush=True)
        tasks=build_tasks(prompts,tok,(256,));assert len(tasks)==len(prompts)*2
        allrows=[];timings=[]
        for i,task in enumerate(tasks):
            status.update(state='running',total=len(tasks),current_prompt=task.prompt['id'],position=task.position)
            save(p/'status.json',status);t=time.monotonic()
            rows,text=run_task(model,tok,ck,task,steps,i==0,20260923+i)
            dt=time.monotonic()-t;timings.append(dt);allrows.extend(rows)
            with (p/'metrics.jsonl').open('a') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            with (p/'continuations.jsonl').open('a') as f:f.write(json.dumps(text)+'\n')
            with (p/'outputs.md').open('a') as f:
                f.write(f"## {task.prompt['id']} / prefix {task.position} ({task.actual_tokens} tokens)\n\n")
                f.write('Reference continuation:\n\n```text\n'+text['reference_text']+'\n```\n\n')
                f.write('| Step | Method | KL | Reference next token | Probe next token |\n|---|---|---:|---|---|\n')
                for r in rows:
                    clean=lambda s:json.dumps(s,ensure_ascii=False).replace('|','\\|')
                    f.write(f"| {r['gen_step']} | {r['method']} | {r['kl']:.6g} | {clean(r['reference_top1_text'])} | {clean(r['probe_top1_text'])} |\n")
                f.write('\n')
            save(p/'summary.json',summaries(allrows))
            avg=sum(timings)/len(timings)
            status.update(completed=i+1,mean_task_seconds=avg,eta_unix=time.time()+avg*(len(tasks)-i-1),updated_unix=time.time())
            save(p/'status.json',status)
            print(f'[{i+1}/{len(tasks)}] {task.prompt["id"]} pos={task.position} T={task.actual_tokens}: {dt:.2f}s',flush=True)
            free_memory()
        summary=summaries(allrows)
        expected=len(prompts)*(steps//4)
        assert len(summary)==2*len(METHODS) and all(r['n']==expected for r in summary)
        with (p/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        status.update(state='completed',finished_unix=time.time());save(p/'status.json',status)
        print('COMPLETED',flush=True)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc(),finished_unix=time.time());save(p/'status.json',status);raise


if __name__=='__main__':main()
