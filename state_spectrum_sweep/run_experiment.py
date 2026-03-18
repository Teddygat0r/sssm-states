from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


MODEL_NAME = "Qwen/Qwen3.5-4B"
TARGET_LAYER_SUBSTR = "30"
SAVE_EVERY = 5  # Set to 2 to save every other token, etc.
TOP_N_SINGULAR_VALUES = 16
MAX_NEW_TOKENS = 100
EXPERIMENTS_ROOT = Path(__file__).parent / "experiments"

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

    deltas = [
        generate_delta(cache_store[i], cache_store[0]) for i in range(1, len(cache_store))
    ]

    if isinstance(deltas[0], list):
        torch_deltas = torch.stack([torch.stack(layer, dim=0) for layer in deltas], dim=0)
    else:
        torch_deltas = torch.stack(deltas, dim=0)

    s = torch.linalg.svdvals(torch_deltas)
    s_energy = get_s_energy(s, n=TOP_N_SINGULAR_VALUES)
    return s_energy, response, len(cache_store), tuple(torch_deltas.shape)


def main():
    run_dir = EXPERIMENTS_ROOT / datetime.now().strftime("suite_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

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

    print(f"Running suite with {len(PROMPT_SUITE)} prompts")
    print(f"Saving outputs to: {run_dir}")

    for i, prompt in enumerate(PROMPT_SUITE, start=1):
        prompt_id = f"prompt_{i:02d}"
        print(f"\n[{i:02d}/{len(PROMPT_SUITE)}] {prompt_id}")
        try:
            s_energy, response, saved_states, deltas_shape = run_single_prompt(
                prompt=prompt, model=model, tokenizer=tokenizer, target_modules=target_modules
            )
            output_path = run_dir / f"{prompt_id}_s_energy.pt"
            torch.save(s_energy.cpu(), output_path)
            print(f"Saved recurrent states: {saved_states}")
            print(f"torch_deltas shape: {deltas_shape}")
            print(f"s_energy shape: {tuple(s_energy.shape)}")
            print(f"Response: {response.strip()}")
            print(f"Saved file: {output_path.name}")
        except Exception as exc:
            print(f"Failed for {prompt_id}: {exc}")

    print("\nDone.")
    print(f"Experiment artifacts are in: {run_dir}")

if __name__ == "__main__":
    main()
