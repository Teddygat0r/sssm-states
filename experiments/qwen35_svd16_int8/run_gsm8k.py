"""Full GSM8K 5-shot, repeated rank-16/INT8 factor compression."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
import torch
from run import ROOT, ARMS, quantize, write_json, kc, randomized_svd_eigh

SOURCE = ROOT / 'eval_results/gsm8k/Qwen__Qwen3.5-4B/samples_gsm8k_2026-04-03T02-02-47.394179.jsonl'
STOP = ['Question:', '</s>', '<|im_end|>']
STATE_INT8 = False


def grade(text, target):
    def normalize(s):
        for pattern in [',', r'\$', r'(?s).*#### ', r'\.$']:
            s = re.sub(pattern, '', s)
        return s.lower()
    strict = re.findall(r'#### (\-?[0-9\.\,]+)', text)
    flex = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', text)
    strict = strict[0].strip() if strict else '[invalid]'
    flex = next((x for x in flex[-1] if x), '[invalid]').strip() if flex else '[invalid]'
    return dict(strict_prediction=strict, flexible_prediction=flex,
                strict_correct=normalize(strict) == normalize(target),
                flexible_correct=normalize(flex) == normalize(target))


def load_questions():
    unique = {}
    for line in SOURCE.open():
        r = json.loads(line)
        prompt = r['arguments']['gen_args_0']['arg_0']
        q = dict(doc_id=r['doc_id'], question=r['doc']['question'], target=r['target'], prompt=prompt)
        assert prompt.count('Question:') == 6
        assert prompt.endswith('Question: ' + q['question'] + '\nAnswer:')
        if q['doc_id'] in unique:
            assert unique[q['doc_id']] == q
        unique[q['doc_id']] = q
    assert len(unique) == 1319
    return [unique[i] for i in sorted(unique)]


def compress_three(cache, indices, seed, ended):
    for row in range(1, len(ARMS)):
        if ended[row]:
            continue
        states = torch.stack([cache.layers[i].recurrent_states[row].float() for i in indices])
        if STATE_INT8:
            restored = quantize(states.flatten(-2), -1).reshape_as(states)
            assert torch.isfinite(restored).all()
            for j, li in enumerate(indices):
                kc.write_recurrent_row(cache, li, row, restored[j])
            continue
        torch.manual_seed(seed)
        u, s, vh = randomized_svd_eigh(states, rank=16, oversample=8, n_iter=2)
        left, right = u*s.sqrt().unsqueeze(-2), s.sqrt().unsqueeze(-1)*vh
        if row == 2:
            left, right = quantize(left, -2), quantize(right, -1)
        restored = left @ right
        assert torch.isfinite(restored).all()
        for j, li in enumerate(indices):
            kc.write_recurrent_row(cache, li, row, restored[j])


@torch.inference_mode()
def generate(model, tok, question, max_tokens, check, heartbeat):
    t0 = time.monotonic()
    ids = tok(question['prompt'], return_tensors='pt').input_ids.cuda()
    out = model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    indices = kc.recurrent_layer_indices(cache)
    assert len(indices) == 24
    n_arms = len(ARMS)
    kc.cache_repeat_(cache, n_arms)
    token = ids[:, -1:].repeat(n_arms, 1)
    tokens = [[] for _ in ARMS]
    texts = ['']*n_arms
    ended = [False]*n_arms
    reasons = ['max_tokens']*n_arms
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    eos.add(tok.eos_token_id)
    for step in range(max_tokens):
        if check and step == 0:
            probe = copy.deepcopy(cache)
            ref = kc.forward_step(model, token, probe, ids.shape[1]-1, 'past_key_values').logits[:, -1].clone()
            assert all(torch.equal(ref[0], ref[j]) for j in range(1, n_arms))
            del probe
            control = [cache.layers[i].recurrent_states[0].clone() for i in indices]
        compress_three(cache, indices, 20260921+question['doc_id']*1000+step, ended)
        if check and step == 0:
            assert all(torch.equal(cache.layers[i].recurrent_states[0], c) for i,c in zip(indices,control))
        out = kc.forward_step(model, token, cache, ids.shape[1]-1+step, 'past_key_values')
        cache = out.past_key_values
        logits = out.logits[:, -1]
        assert torch.isfinite(logits).all()
        if check and step == 0:
            assert torch.equal(ref[0], logits[0]), 'Baseline contamination'
            print('CHECKS PASSED: baseline batch equality, untouched control, finite states/logits', flush=True)
        nxt = logits.argmax(-1)
        for j, t in enumerate(nxt.tolist()):
            if ended[j]:
                continue
            tokens[j].append(t)
            text = tok.decode(tokens[j], skip_special_tokens=False)
            stops = [text.index(s) for s in STOP if s in text]
            if stops:
                text = text[:min(stops)]
                ended[j], reasons[j] = True, 'stop_string'
            elif t in eos:
                text = tok.decode(tokens[j][:-1], skip_special_tokens=False)
                ended[j], reasons[j] = True, 'eos'
            texts[j] = text
        if step == 3 or (step+1) % 64 == 0:
            heartbeat(step+1, time.monotonic()-t0)
        if all(ended):
            break
        token = nxt.reshape(n_arms, 1)
    return dict(**question, seconds=time.monotonic()-t0, prompt_tokens=ids.shape[1],
                decode_steps=step+1, arms={a: dict(text=texts[j], token_ids=tokens[j],
                stopped=reasons[j], **grade(texts[j], question['target'])) for j,a in enumerate(ARMS)})


def main():
    global ARMS, STATE_INT8
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=768)
    ap.add_argument('--state-int8', action='store_true', help='Full-state symmetric INT8 vs baseline; no SVD')
    args = ap.parse_args()
    STATE_INT8 = args.state_int8
    if STATE_INT8:
        ARMS = ['uncompressed', 'int8_state']
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    status = dict(state='starting', started_unix=time.time(), completed=0)
    write_json(outdir/'status.json', status)
    try:
        torch.set_num_threads(8)
        torch.manual_seed(20260921)
        questions = load_questions()
        if args.limit:
            questions = questions[:args.limit]
        write_json(outdir/'questions.json', questions)
        write_json(outdir/'config.json', dict(model='Qwen/Qwen3.5-4B', arms=ARMS, num_fewshot=5,
                   max_tokens=args.max_tokens, num_questions=len(questions), greedy=True, stop=STOP,
                   source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   seed=20260921, rank=None if STATE_INT8 else 16, bits=8,
                   oversample=None if STATE_INT8 else 8, n_iter=None if STATE_INT8 else 2,
                   quantization=('full recurrent matrix; symmetric signed INT8 [-127,127] per layer/head; FP32 scales; no SVD'
                   if STATE_INT8 else 'L=U sqrt(S), R=sqrt(S) Vh; symmetric INT8 per rank component; FP32 scales'),
                   compression='before last prompt token and every decode step, all recurrent layers; dense reconstruction',
                   grading='GSM8K lm-eval v3.0 regex extraction and exact-match normalization',
                   torch=torch.__version__, gpu=torch.cuda.get_device_name()))
        from run_reconstruction import load_qwen35
        print(f'Loading model for {len(questions)} questions', flush=True)
        model, tok, _, _ = load_qwen35()
        records = []
        for q in questions:
            status.update(state='running', total=len(questions), current_doc_id=q['doc_id'])
            write_json(outdir/'status.json', status)
            def heartbeat(steps, seconds):
                status.update(current_decode_steps=steps, current_seconds=seconds, updated_unix=time.time())
                write_json(outdir/'status.json', status)
                if not records and steps == 4:
                    print(f'STARTUP VERIFIED: 4 decode steps in {seconds:.2f}s', flush=True)
            r = generate(model, tok, q, args.max_tokens, not records, heartbeat)
            with (outdir/'samples.jsonl').open('a') as f:
                f.write(json.dumps(r)+'\n')
            records.append(r)
            summary = {a: dict(n=len(records),
                strict_accuracy=sum(x['arms'][a]['strict_correct'] for x in records)/len(records),
                flexible_accuracy=sum(x['arms'][a]['flexible_correct'] for x in records)/len(records),
                truncated=sum(x['arms'][a]['stopped']=='max_tokens' for x in records)) for a in ARMS}
            write_json(outdir/'summary.json', summary)
            avg = sum(x['seconds'] for x in records)/len(records)
            status.update(completed=len(records), mean_question_seconds=avg,
                eta_unix=time.time()+avg*(len(questions)-len(records)), updated_unix=time.time())
            write_json(outdir/'status.json', status)
            print(f"[{len(records)}/{len(questions)}] {r['seconds']:.1f}s, {r['decode_steps']} steps; remaining {avg*(len(questions)-len(records))/3600:.2f}h", flush=True)
        with (outdir/'outputs.md').open('w') as f:
            for r in records:
                f.write(f"## Question {r['doc_id']}\n\n{r['question']}\n\n")
                for a in ARMS:
                    f.write(f"### {a} (correct={r['arms'][a]['strict_correct']})\n\n{r['arms'][a]['text']}\n\n")
        status.update(state='completed', finished_unix=time.time())
        write_json(outdir/'status.json', status)
        print('COMPLETED', flush=True)
    except BaseException:
        status.update(state='failed', error=traceback.format_exc(), finished_unix=time.time())
        write_json(outdir/'status.json', status)
        raise


if __name__ == '__main__':
    main()
