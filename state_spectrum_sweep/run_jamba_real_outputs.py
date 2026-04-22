from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
DEFAULT_MODEL = "jamba-reasoning-3b"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50

PROMPT_SUITE = [
    "Hi there! Who are you?",
    "Explain photosynthesis in one paragraph.",
    "Write three creative names for a coffee shop.",
    "What is the capital of Japan and one famous landmark there?",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Translate this to Spanish: I enjoy learning new things every day.",
    "Give me a short bedtime story about a robot and a cat.",
    "List five practical ways to reduce household energy usage.",
    "What are the differences between lists and tuples in Python?",
    "Write a haiku about rain in a city.",
    "Create a two-day itinerary for visiting New York City.",
    "Explain Newton's second law with a simple example.",
    "Suggest a healthy breakfast under 400 calories.",
    "Draft a polite email asking for a project deadline extension.",
    "What are the main ideas behind gradient descent?",
    "Give me three interview questions for a junior data scientist role.",
    "Describe the plot of Romeo and Juliet in four sentences.",
    "Write a short dialogue between a teacher and a curious student.",
    "Provide a regex for validating a basic email format.",
    "What is overfitting in machine learning, and how can we reduce it?",
    "Generate a list of ten random words and use each in a sentence.",
    "Explain recursion to a 10-year-old.",
    "Compare REST and GraphQL in a concise table-style format.",
    "Write a simple Python function to check if a number is prime.",
    "What are three ethical concerns with large language models?",
    "Give me a 7-day beginner workout plan with light equipment.",
    "Describe how rainbows form using simple physics.",
    "Write a motivational message for someone learning to code.",
    "Explain the difference between precision and recall.",
    "Create a short sci-fi scene set on a lunar research station.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 30 prompts through an untouched Jamba model and save the raw outputs."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Optional tokenizer id/path. Defaults to --model.",
    )
    parser.add_argument(
        "--prompts-file",
        default="",
        help="Optional .txt, .json, or .jsonl prompt file. If omitted, uses the built-in 30-prompt suite.",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=30,
        help="Number of prompts to run. Default: 30.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate per prompt.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. If omitted, generation is greedy and deterministic.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature when --do-sample is enabled.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Top-p value when --do-sample is enabled.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-k value when --do-sample is enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --do-sample is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to a timestamped folder under state_spectrum_sweep/experiments.",
    )
    return parser


def _dtype_for_runtime() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _load_prompts(prompts_file: str) -> list[str]:
    if not prompts_file:
        return list(PROMPT_SUITE)

    path = Path(prompts_file)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [prompt for prompt in prompts if prompt]

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            prompts: list[str] = []
            for item in data:
                if isinstance(item, str):
                    prompts.append(item)
                elif isinstance(item, dict) and "prompt" in item:
                    prompts.append(str(item["prompt"]))
                else:
                    raise ValueError("JSON prompt entries must be strings or objects with a `prompt` field.")
            return prompts
        raise ValueError("JSON prompt file must contain a list.")

    if suffix == ".jsonl":
        prompts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and "prompt" in item:
                prompts.append(str(item["prompt"]))
            else:
                raise ValueError("JSONL prompt entries must be strings or objects with a `prompt` field.")
        return prompts

    raise ValueError("Unsupported prompts file. Use .txt, .json, or .jsonl.")


def _prompt_to_model_text(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


def _resolve_model_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for module_name in ("model.embed_tokens", "backbone.embeddings", "model.embeddings", "embeddings", ""):
            device_name = hf_device_map.get(module_name)
            if device_name not in (None, "disk"):
                return torch.device(device_name)
        for device_name in hf_device_map.values():
            if device_name not in (None, "disk"):
                return torch.device(device_name)
    return next(model.parameters()).device


def _make_generation_kwargs(args: argparse.Namespace, tokenizer) -> dict:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    else:
        kwargs.update(do_sample=False)
    return kwargs


def main() -> None:
    args = build_parser().parse_args()

    prompts = _load_prompts(args.prompts_file)
    if args.prompt_limit > 0:
        prompts = prompts[: args.prompt_limit]
    if not prompts:
        raise ValueError("No prompts to run.")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else EXPERIMENTS_ROOT / datetime.now().strftime("jamba_real_outputs_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer_name = args.tokenizer or args.model
    dtype = _dtype_for_runtime()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()

    if args.do_sample:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    input_device = _resolve_model_device(model)
    generation_kwargs = _make_generation_kwargs(args, tokenizer)
    results = []
    prompt_id_width = max(2, len(str(len(prompts))))

    print(f"Model: {args.model}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"Sampling: {'on' if args.do_sample else 'off'}")
    print(f"Saving outputs to: {output_dir}")

    for index, prompt in enumerate(prompts, start=1):
        prompt_id = f"prompt_{index:0{prompt_id_width}d}"
        print(f"\n[{index:0{prompt_id_width}d}/{len(prompts):0{prompt_id_width}d}] {prompt_id}")

        model_text = _prompt_to_model_text(tokenizer, prompt)
        model_inputs = tokenizer([model_text], return_tensors="pt").to(input_device)

        with torch.inference_mode():
            generated_ids = model.generate(**model_inputs, **generation_kwargs)

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        result = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "response": response,
            "model": args.model,
            "tokenizer": tokenizer_name,
            "prompt_tokens": int(model_inputs.input_ids.shape[1]),
            "generated_tokens": int(output_ids.shape[0]),
        }
        results.append(result)

        (output_dir / f"{prompt_id}_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        (output_dir / f"{prompt_id}_response.txt").write_text(response + "\n", encoding="utf-8")
        (output_dir / f"{prompt_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        print(f"Prompt: {prompt}")
        print(f"Generated tokens: {result['generated_tokens']}")
        print(f"Response: {response}")

    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in results),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Saved aggregate results to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
