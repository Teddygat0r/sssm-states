# Qwen3.5-4B rank-16 + INT8 recurrent-state quality experiment

Run `./experiments/qwen35_svd16_int8/launch.sh full` from the repository.
The launcher creates a detached systemd user service; `smoke` uses the same
launcher and code with one task and four decode steps. The user service survives
the conversation ending; it does not survive reboot or termination of the user
service manager (this host currently has lingering disabled).

## Protocol

- The repository's `svd_sweep._helpers.load_prompts` supplies 10 ShareGPT chunks
  and 10 code chunks, including its documented dataset fallback behavior.
  Logs record the actual source; the exact prompt text is saved in `prompts.json`.
- Each chunk is evaluated at 256 and 2048 tokens: 40 tasks total.
- Three conditions: native uncompressed state, randomized rank-16 SVD,
  randomized rank-16 SVD followed by symmetric signed INT8 factor quantization.
- SVD uses the repository's `randomized_svd_eigh`, oversampling 8, two power
  iterations. Seeds are fixed and shared across arms to avoid changing the sketch
  just because quantization is enabled.
- Factorization is `L = U sqrt(S)`, `R = sqrt(S) Vh`. Each column of L and each
  row of R has its own FP32 max-absolute-value scale per head/layer. Values are
  rounded and clipped to [-127, 127], stored as INT8, then dequantized.
- Every recurrent layer is compressed. Attention KV, convolution state and
  model weights retain their original precision. Reconstructed dense states
  are used for model computation: this is a quality simulation, not a compressed
  kernel performance or actual resident-memory benchmark.
- Prefill the first T-1 context tokens, compress, then consume the last context
  token to score/generate the first continuation token. Recompress before every
  subsequent decode step. This tests repeated compression, not only a one-time
  prompt-state perturbation.
- Three batch rows teacher-force the SAME 128 reference tokens from the source
  chunk. Report token NLL, perplexity, KL(full || compressed), top-1 agreement.
  Three independent batch rows generate greedy continuations (up to 128 tokens,
  displayed only through first EOS). KL is only computed on matched reference
  contexts, never between divergent generated histories.
- These are raw text completions consistent with the existing sweep, not chat
  templated instruction benchmarks. Perplexity and KL indicate output fidelity;
  they do not establish task accuracy. The sample is a pilot, not an exhaustive
  benchmark. Summary means give equal weight to every 128-token task.

## Files and status

`latest_full.txt` points to the current results directory, containing:

- `config.json`: experiment settings and runner SHA256.
- `prompts.json`: exact input chunks.
- `samples.jsonl`: incrementally saved metrics, prompt/reference token IDs, and
  generated text/token IDs for all three arms.
- `status.json`: starting/running/completed/failed, task progress, projected ETA,
  failure traceback if applicable; updated atomically after each task.
- `run.log`, `unit.txt`: service output and service name.
- `summary.json`, `outputs.md`: final aggregate metrics and readable continuations.

Inspect once with `systemctl --user status UNIT` and `cat OUTPUT/status.json`.
Use `systemctl --user stop UNIT` to stop. No automatic retry or resume is enabled;
failed runs retain completed samples and an error status. Each launch uses a new
directory. Initial runtime estimates are based on the first completed task and
may shift with context length, numerical fallbacks, or GPU contention.
