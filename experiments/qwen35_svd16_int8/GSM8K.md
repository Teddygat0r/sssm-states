# GSM8K repeated-compression evaluation

Launch: `./experiments/qwen35_svd16_int8/launch_gsm8k.sh full`.
`smoke` runs the same launcher with one question and the full 768-token budget.
Both are detached systemd user services. `latest_gsm8k_full.txt` records the
results directory. Jobs survive the conversation ending, but not reboot.

The run evaluates all 1,319 test questions with the exact cached 5-shot prompts
from the repository's April 3 GSM8K lm-eval sample dump (source path and SHA256
saved in config.json). It does not reuse model predictions. All examples in each
prompt, prompt suffixes, duplicate records, and total question count are checked.

Conditions are native state, rank-16 randomized SVD, and rank-16 SVD with INT8
factors. Factor quantization matches README.md: balanced sqrt(S) factors,
symmetric per-component scales, FP32 metadata, oversampling 8, two power
iterations. Every recurrent layer is compressed before consuming the last prompt
token and before every subsequent decode step. Attention, convolution states,
and model weights are unchanged. Dense reconstruction simulates quality, not
compressed-kernel memory usage or throughput. This repeated-compression protocol
differs from the older ablations/GSM8k experiment's prefill-only compression.

Three independent batch rows greedily decode, with a maximum of 768 tokens per
answer. Rows stop recording at the first GSM8K stop string or EOS. Finished rows
may continue internal computation until all rows finish; their saved answers do
not change. No chat template or additional instruction is added.

Strict and flexible answer extraction and exact-match normalization implement
installed lm-eval's GSM8K v3.0 YAML. Strict extraction takes the first `#### N`,
flexible extraction takes the last numeric match. Missing or wrong answers count
as incorrect, including truncations; truncation counts are separately reported.
The scorer was checked against the existing saved lm-eval scores.

Files:
- config.json and questions.json: protocol, provenance and exact prompts.
- status.json: progress, current decode heartbeat, ETA, completion/error status.
- samples.jsonl: incremental question, answers, generated token IDs and grades.
- summary.json: running strict/flexible accuracy and truncation counts per arm.
- outputs.md: readable final answers, written when all questions complete.
- run.log and unit.txt: launcher output and systemd service name.

Runtime ETA is updated after each question; early estimates may vary considerably
with answer length. No automatic restart or resume is enabled. Failures retain
completed samples and traceback. Use `systemctl --user stop UNIT` to stop.

## Full-state INT8 follow-up

Run `./experiments/qwen35_svd16_int8/launch_gsm8k_state_int8.sh full`.
The same runner's `--state-int8` mode compares two fresh batch rows:
`uncompressed` and `int8_state`, over all 1,319 questions. `smoke` completes
one question using the same 768-token budget. The pointer file is
`latest_gsm8k_state_int8_full.txt`.

There is no SVD in this mode. Each full recurrent matrix is rounded to signed
INT8 [-127,127], with one max-absolute-value FP32 scale per layer/head, and then
dequantized for dense computation. This uses the same symmetric quantization
rule as the factor experiment, now applied to full matrices rather than factors.
It differs from the old `ablations/GSM8k` affine/min-max q8 quantizer. All prompt,
generation, grading, and repeated-compression settings remain as above. The
native baseline is rerun with the same two-row batch shape as the INT8 arm.

Synthetic validation checks that SVD is never invoked, control states stay
unchanged, zero states remain finite, and rounding error is bounded by half a
quantization step. The first real question also checks baseline logit isolation.
