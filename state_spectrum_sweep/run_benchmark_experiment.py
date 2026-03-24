from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


MODEL_NAME = "Qwen/Qwen3.5-4B"
TARGET_LAYER_SUBSTR = "30"
SAVE_EVERY = 5  # Set to 2 to save every other token, etc.
TOP_N_SINGULAR_VALUES = 16
MAX_NEW_TOKENS = 200
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"
MMLU_DATASET_NAME = "cais/mmlu"
MMLU_DATASET_CONFIG = "all"
MMLU_SPLITS = ("auxiliary_train", "dev", "validation", "test")

def _clone_recurrent_state_to_cpu(recurrent_state):
    if recurrent_state is None:
        return None

    if isinstance(recurrent_state, torch.Tensor):
        return recurrent_state.detach().to("cpu").clone()

    if isinstance(recurrent_state, (list, tuple)):
        return [
            state.detach().to("cpu").clone() if state is not None else None
            for state in recurrent_state
        ]

    raise TypeError(f"Unsupported recurrent_state type: {type(recurrent_state)}")


def _format_mmlu_prompt(example):
    choices = "\n".join(
        f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(example["choices"])
    )
    return (
        f"Subject: {example['subject']}\n\n"
        f"Question: {example['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Answer with the single best option."
    )


def load_mmlu_prompts():
    dataset = load_dataset(MMLU_DATASET_NAME, MMLU_DATASET_CONFIG)
    prompts = []
    for split in MMLU_SPLITS:
        if split not in dataset:
            continue
        for idx, example in enumerate(dataset[split]):
            prompts.append(
                {
                    "id": f"{split}_{idx:05d}",
                    "prompt": _format_mmlu_prompt(example),
                    "split": split,
                }
            )
    if not prompts:
        raise RuntimeError("No prompts found in MMLU dataset.")
    return prompts


def get_post_hook(cache_store, save_every=1):
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    step_counter = {"n": 0}

    def post_hook(module, args, kwargs, output):
        cache_params = kwargs.get("cache_params", None)
        if cache_params is None:
            return

        if step_counter["n"] % save_every == 0:
            recurrent_state = getattr(cache_params, "recurrent_state", None)
            if recurrent_state is None:
                # Backward-compatible fallback depending on model implementation.
                recurrent_state = getattr(cache_params, "recurrent_states", None)
            cache_store.append(_clone_recurrent_state_to_cpu(recurrent_state))

        step_counter["n"] += 1

    return post_hook


def generate_delta(state, state_ref):
    if isinstance(state, torch.Tensor):
        return state - state_ref

    if isinstance(state, list):
        return [
            state[i] - state_ref[i]
            for i in range(len(state))
            if state[i] is not None and state_ref[i] is not None
        ]

    raise TypeError(f"Unsupported state type for delta: {type(state)}")


def get_s_energy(singular_values, n=-1):
    s_energy = torch.sum(singular_values**2, dim=-1, keepdim=True)
    s_nume = singular_values**2
    if n == -1:
        return torch.sum(s_nume, dim=-1, keepdim=True) / s_energy
    return torch.sum(s_nume[..., :n], dim=-1, keepdim=True) / s_energy

# def get_kl_divergence(U, s, Vh, module, recurrent_state, conv_state):
    

def run_single_prompt(prompt, model, tokenizer, target_modules):
    cache_store = []
    hook_handles = []

    for module in target_modules:
        handle = module.register_forward_hook(
            get_post_hook(cache_store, save_every=SAVE_EVERY),
            with_kwargs=True,
        )
        hook_handles.append(handle)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    for handle in hook_handles:
        handle.remove()

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    if len(cache_store) < 2:
        raise ValueError(
            "Need at least 2 saved recurrent states to compute deltas. "
            "Increase MAX_NEW_TOKENS or reduce SAVE_EVERY."
        )

    deltas = [generate_delta(cache_store[i], cache_store[0]) for i in range(1, len(cache_store))]

    if isinstance(deltas[0], list):
        torch_deltas = torch.stack([torch.stack(layer, dim=0) for layer in deltas], dim=0)
    else:
        torch_deltas = torch.stack(deltas, dim=0)

    s = torch.linalg.svdvals(torch_deltas)
    s_energy = get_s_energy(s, n=TOP_N_SINGULAR_VALUES)
    return s_energy, response, len(cache_store), tuple(torch_deltas.shape)


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("mmlu_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    mmlu_prompts = load_mmlu_prompts()
    print(f"Loaded {len(mmlu_prompts)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print(f"Loaded model")

    target_modules = [
        module
        for name, module in model.named_modules()
        if isinstance(module, Qwen3_5GatedDeltaNet) and TARGET_LAYER_SUBSTR in name
    ]
    if not target_modules:
        raise RuntimeError(
            "No Qwen3_5GatedDeltaNet module matched TARGET_LAYER_SUBSTR. "
            "Adjust TARGET_LAYER_SUBSTR."
        )

    print(f"Running MMLU benchmark with {len(mmlu_prompts)} prompts")
    print(f"Saving outputs to: {run_dir}")

    for i, prompt_entry in enumerate(mmlu_prompts, start=1):
        prompt_id = prompt_entry["id"]
        prompt = prompt_entry["prompt"]
        print(f"\n[{i:05d}/{len(mmlu_prompts):05d}] {prompt_id}")
        try:
            s_energy, response, saved_states, deltas_shape = run_single_prompt(
                prompt=prompt, model=model, tokenizer=tokenizer, target_modules=target_modules
            )
            tensor_output_path = run_dir / f"{prompt_id}_s_energy.pt"
            text_output_path = run_dir / f"{prompt_id}_prompt_response.txt"
            torch.save(s_energy.cpu(), tensor_output_path)
            text_output_path.write_text(
                f"PROMPT_ID: {prompt_id}\n"
                f"SPLIT: {prompt_entry['split']}\n\n"
                f"PROMPT:\n{prompt}\n\n"
                f"RESPONSE:\n{response.strip()}\n",
                encoding="utf-8",
            )
            print(f"Saved recurrent states: {saved_states}")
            print(f"torch_deltas shape: {deltas_shape}")
            print(f"s_energy shape: {tuple(s_energy.shape)}")
            print(f"Response: {response.strip()}")
            print(f"Saved files: {tensor_output_path.name}, {text_output_path.name}")
        except Exception as exc:
            print(f"Failed for {prompt_id}: {exc}")

    print("\nDone.")
    print(f"Experiment artifacts are in: {run_dir}")

if __name__ == "__main__":
    main()
