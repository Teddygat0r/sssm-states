import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

from HFLM_svd import HFLM_svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM8K with lm-eval using custom HFLM_svd wrapper."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="HF model name or path.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gsm8k",
        help="lm-eval task name (default: gsm8k).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results",
        help="Directory for result json output.",
    )
    parser.add_argument(
        "--rank-k",
        type=int,
        default=64,
        help="Low-rank SVD rank for recurrent state approximation.",
    )
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=128,
        help="Prefix length setting for HFLM_svd.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to HFLM_svd (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="1",
        help="Batch size passed to lm-eval model wrapper.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional sample limit for quick testing.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=5,
        help="Few-shot count for GSM8K.",
    )
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=512,
        help="Maximum generated tokens per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature. 0.0 is greedy decoding.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply tokenizer chat template in lm-eval.",
    )
    return parser.parse_args()


def _resolve_task_name(task_manager: TaskManager, requested_task: str) -> str:
    matches = task_manager.match_tasks([requested_task])
    if matches:
        return matches[0]

    known = task_manager.all_tasks
    gsm_like = [t for t in known if "gsm8k" in t.lower()]
    hint = ", ".join(gsm_like[:10]) if gsm_like else "No gsm8k-like tasks found."
    raise ValueError(
        f"Task '{requested_task}' was not found by lm-eval TaskManager. "
        f"Available gsm8k-like tasks: {hint}"
    )


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, np.dtype):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size
    lm = HFLM_svd(
        pretrained=args.model,
        backend="causal",
        trust_remote_code=True,
        device=args.device,
        batch_size=batch_size,
        rank_k=args.rank_k,
        prefix_length=args.prefix_length,
    )

    task_manager = TaskManager()
    task_name = _resolve_task_name(task_manager, args.task)

    gen_kwargs = {
        "max_gen_toks": args.max_gen_toks,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0.0,
    }

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        task_manager=task_manager,
        num_fewshot=args.num_fewshot,
        batch_size=batch_size,
        limit=args.limit,
        bootstrap_iters=0,
        gen_kwargs=gen_kwargs,
        log_samples=True,
        apply_chat_template=args.apply_chat_template,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_task = task_name.replace("/", "_")
    out_path = output_dir / f"gsm8k_hflm_svd_{safe_task}_{stamp}.json"
    safe_results = _json_safe(results)
    out_path.write_text(
        json.dumps(safe_results, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Finished evaluation for task: {task_name}")
    print(f"Saved results to: {out_path}")

    if results and "results" in results and task_name in results["results"]:
        print("Task metrics:")
        for k, v in results["results"][task_name].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
