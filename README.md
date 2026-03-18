# sssm-states

Local Qwen runner using Hugging Face Transformers.

## Quick start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e .
```

3. Run the default model (`Qwen/Qwen2.5-3B-Instruct`):

```bash
python3 main.py --prompt "Explain state machines in simple terms."
```

## Model selection

Change model id with `--model`:

```bash
python3 main.py --model "Qwen/Qwen2.5-3B-Instruct" --prompt "Hello!"
```

If you specifically want a different "Qwen 3.5B" variant, pass that exact Hugging Face model id via `--model`.

## Notes

- First run downloads model weights from Hugging Face and can take several minutes.
- On GPU, the script uses fp16 automatically. On CPU, it falls back to fp32.
- If memory is tight, reduce `--max-new-tokens`.
# sssm-states