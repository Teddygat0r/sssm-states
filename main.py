import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a prompt with a Qwen model.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="Hugging Face model id (default: Qwen/Qwen3.5-4B).",
    )
    parser.add_argument(
        "--prompt",
        default="Write a short haiku about open-source software.",
        help="Prompt text to send to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(response.strip())


if __name__ == "__main__":
    main()
