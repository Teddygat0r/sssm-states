import os
import sys
import json
import re
import time
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

HERE = Path(__file__).resolve().parent
model_map = json.loads((HERE / "config/model2path.json").read_text(encoding="utf-8"))
maxlen_map = json.loads((HERE / "config/model2maxlen.json").read_text(encoding="utf-8"))
template_0shot = (HERE / "prompts/0shot.txt").read_text(encoding="utf-8")
template_0shot_cot = (HERE / "prompts/0shot_cot.txt").read_text(encoding="utf-8")
template_0shot_cot_ans = (HERE / "prompts/0shot_cot_ans.txt").read_text(encoding="utf-8")
template_no_context = (HERE / "prompts/0shot_no_context.txt").read_text(encoding="utf-8")
template_rag = (HERE / "prompts/0shot_rag.txt").read_text(encoding="utf-8")


def load_v2_dataset(local_json=None):
    if local_json is not None:
        path = local_json
    else:
        path = hf_hub_download(repo_id="THUDM/LongBench-v2", filename="data.json", repo_type="dataset")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_and_tokenizer(model_key, svd_rank=None, svd_interval=1024):
    path = model_map[model_key]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if "qwen3.5" in model_key:
        from modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
        model = Qwen3_5MoeForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        if svd_rank is not None:
            for layer in model.model.layers:
                if getattr(layer, "layer_type", None) == "linear_attention":
                    layer.linear_attn.svd_rank = svd_rank
                    layer.linear_attn.svd_interval = svd_interval
                    layer.linear_attn.svd_step_counter = 0
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        ).eval()
    return model, tokenizer


def middle_truncate(prompt, tokenizer, max_len):
    ids = tokenizer.encode(prompt)
    if len(ids) <= max_len:
        return prompt
    half = max_len // 2
    ids = ids[:half] + ids[-half:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def build_prompt(item, context, template):
    return (
        template
        .replace("$DOC$", context.strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def query_llm(prompt, model, tokenizer, max_new_tokens, temperature):
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat], return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def extract_answer(response):
    response = response.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", response)
    if m:
        return m.group(1)
    m = re.search(r"The correct answer is ([A-D])", response)
    if m:
        return m.group(1)
    return None


def strip_thinking(response):
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()
    return response


def run(args):
    os.makedirs(args.save_dir, exist_ok=True)
    suffix = ""
    if args.rag > 0:
        suffix = f"_rag_{args.rag}"
    elif args.no_context:
        suffix = "_no_context"
    elif args.cot:
        suffix = "_cot"
    if args.svd_rank is not None:
        suffix += f"_svd{args.svd_rank}-i{args.svd_interval}"
    out_file = os.path.join(args.save_dir, f"{args.model}{suffix}.jsonl")

    data_all = load_v2_dataset()
    data_all = [
        {k: item[k] for k in (
            "_id", "domain", "sub_domain", "difficulty", "length", "question",
            "choice_A", "choice_B", "choice_C", "choice_D", "answer", "context"
        )}
        for item in data_all
    ]

    # Filter-by previously written IDs for resumability.
    seen = set()
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["_id"])
                except Exception:
                    pass
    data = [d for d in data_all if d["_id"] not in seen]

    if args.difficulty:
        data = [d for d in data if d["difficulty"] == args.difficulty]
    if args.length:
        data = [d for d in data if d["length"] == args.length]
    if args.num_samples is not None:
        data = data[: args.num_samples]

    print(f"running {len(data)} samples -> {out_file}")
    if not data:
        return

    model, tokenizer = load_model_and_tokenizer(
        args.model, svd_rank=args.svd_rank, svd_interval=args.svd_interval
    )
    max_len = maxlen_map[args.model]

    with open(out_file, "a", encoding="utf-8") as fout:
        for item in tqdm(data):
            t0 = time.perf_counter()
            context = item["context"]
            if args.rag > 0:
                retrieved = item.get("retrieved_context", [])[: args.rag]
                retrieved = sorted(retrieved, key=lambda x: x["c_idx"])
                context = "\n\n".join(
                    f"Retrieved chunk {i+1}: {x['content']}" for i, x in enumerate(retrieved)
                )
                template = template_rag
            elif args.no_context:
                template = template_no_context
            elif args.cot:
                template = template_0shot_cot
            else:
                template = template_0shot

            prompt = build_prompt(item, context, template)
            prompt = middle_truncate(prompt, tokenizer, max_len)

            if args.cot:
                cot_out = query_llm(prompt, model, tokenizer, max_new_tokens=1024, temperature=0.1)
                cot_out = strip_thinking(cot_out).strip()
                item["response_cot"] = cot_out
                ans_prompt = (
                    template_0shot_cot_ans
                    .replace("$DOC$", context.strip())
                    .replace("$Q$", item["question"].strip())
                    .replace("$C_A$", item["choice_A"].strip())
                    .replace("$C_B$", item["choice_B"].strip())
                    .replace("$C_C$", item["choice_C"].strip())
                    .replace("$C_D$", item["choice_D"].strip())
                    .replace("$COT$", cot_out)
                )
                ans_prompt = middle_truncate(ans_prompt, tokenizer, max_len)
                raw = query_llm(ans_prompt, model, tokenizer, max_new_tokens=args.max_gen, temperature=0.1)
            else:
                raw = query_llm(prompt, model, tokenizer, max_new_tokens=args.max_gen, temperature=0.1)

            response = strip_thinking(raw).strip()
            item["response_raw"] = raw
            item["response"] = response
            item["pred"] = extract_answer(response)
            item["judge"] = item["pred"] == item["answer"]
            item["context"] = context[:1000]
            item["elapsed_seconds"] = time.perf_counter() - t0
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="qwen3.5-moe")
    parser.add_argument("--cot", "-cot", action="store_true")
    parser.add_argument("--no_context", "-nc", action="store_true")
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--max_gen", type=int, default=128, help="Tokens to generate for the answer step.")
    parser.add_argument("--num_samples", type=int, default=None, help="Cap on samples to run; None = all.")
    parser.add_argument("--difficulty", type=str, default=None, choices=[None, "easy", "hard"])
    parser.add_argument("--length", type=str, default=None, choices=[None, "short", "medium", "long"])
    parser.add_argument("--svd_rank", type=int, default=None, help="Low-rank k for GatedDeltaNet SVD compression (None disables).")
    parser.add_argument("--svd_interval", type=int, default=1024, help="Decode tokens between SVD compressions.")
    args = parser.parse_args()
    run(args)
