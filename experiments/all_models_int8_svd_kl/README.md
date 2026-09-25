# Five-model INT8 vs SVD single-step output KL

Launch `./experiments/all_models_int8_svd_kl/launch.sh`. The detached systemd
service first prepares a shared set of 100 ShareGPT and 100 code chunks using
the repository loader (including its public CodeParrot fallback), then smoke-tests
all five models, then runs all five full evaluations sequentially. Any failure
stops the suite with a traceback; completed outputs remain saved. No automatic
retry, resume, or agent monitoring is scheduled. Service survives conversation
end, not reboot. `latest.txt` gives the run directory.

## Protocol

Models: DeltaNet-1.3B, GatedDeltaNet-1.3B, Qwen3.5-4B, Mamba2-1.3B, and
Nemotron-3-Nano-4B, loaded through the existing reconstruction model adapters.

Each prompt independently starts a 128-token greedy reference continuation from
(a) the first 256 prompt tokens and (b) the full prompt, capped at 65,536 tokens
by the original task builder. Exact prompt text and actual token counts are saved.
EOS does not stop the fixed-length reference, matching the original KL sweep.

Every four reference decoding steps, seven independent probe rows receive:

1. **INT8:** affine min/max quantization, one FP32 scale and zero point per
   recurrent layer across all heads. Values are stored as UINT8 codes 0..255 and
   dequantized. This matches the original `run_experiment_quant.py` arithmetic
   in FP32; it is not the recent symmetric per-head or per-column GSM8K quantizer.
2. **SVD:** ranks 1, 2, 4, 8, 16, and 32, using two power iterations and
   floating-point factors without quantization. Sketch groups match the historical
   sweeps: ranks 4/8/16 share a rank-16 sketch with oversampling 8; ranks 1/2/32
   share a rank-32 sketch with oversampling 0. Each rank has its own probe row.

The native row and all seven probe rows consume the same reference token. All
recurrent heads/layers are replaced simultaneously. Attention KV and convolution
states remain identical. Compute KL(native || probe), then discard all seven probe
rows and continue from the native cache only. All methods see exactly the same
reference states, contexts and batch shape. There is no accumulated compressed
rollout drift. There are 6,400 measurements per model/method/prefix condition,
448,000 total metric records, and 2,000 reference continuations.

As in the earlier runner, gen_step 0 consumes the first generated token and
compares distributions predicting generated-token index 1 (zero-based). Saved
`predicted_generation_index` makes this explicit.

## Validation and timing

The real-model smoke phase covers both prompt distributions and both prefix
conditions (8 steps each). For each model, the first task verifies native batch
row equality and that compressed probe rows cannot change native-row logits
(tolerance 1e-3). Every task requires recurrent states, finite reconstructed states
and logits, finite KL, and correct output counts. The native cache is selected
back to batch one after every probe. SVD and affine quantization errors are logged.

Estimated initial runtime for all ranks: 6–14 hours, based on approximately 6.2 hours for the
historical five-model sweep, with uncertainty for prompt lengths, model loading,
compilation, and contention. Each worker updates its own ETA after each task.
The suite status identifies which worker is current. All smoke tests must pass
before full evaluation begins.

## Outputs

At suite root:
- `prompts.json`, `config.json`: shared data, settings, hashes.
- `status.json`, `run.log`, `unit.txt`: phase, model, progress, service status.
- `summary.csv` / `summary.json`: combined model results, updated after each model.
- `comparison.md`: final readable comparison table.

Under `full/<model>/` (and separately `smoke/<model>/`):
- `metrics.jsonl`: per-probe KL, state relative squared Frobenius error, top-1
  agreement, native/probe next-token text and IDs, probe top-5 probabilities.
- `continuations.jsonl`: native reference continuation text/IDs and prompt IDs.
- `outputs.md`: readable continuations and per-step INT8/SVD next-token comparisons.
- `summary.csv` / `summary.json`: mean, median, p90, p99, maximum KL (nats),
  top-1 agreement and state error, separated by prefix/method.
- `status.json`, `config.json`, `run.log`: worker progress, protocol, validation.

Probe predictions are single-step alternatives on a reference history, not
independently generated compressed answers. The methods have different storage
budgets; this is a quality comparison, not an equal-memory or throughput test.

## Parallel model processes

`./experiments/all_models_int8_svd_kl/launch_parallel.sh` reuses the shared prompt
file from the run named by `latest.txt`, creates a new output directory, and runs
all five models in separate Python/CUDA processes concurrently on the same GPU.
Five supervisor threads only manage subprocess lifecycles; no model is shared
between processes. CPU BLAS/PyTorch thread counts remain one per worker.

The actual launcher first runs all five smoke tests concurrently. Full evaluation
starts only after every smoke test succeeds. The full phase then starts all five
workers together. If a model fails, the other models in that phase can finish;
the suite records errors and does not report overall completion. Stopping the
systemd service terminates the supervisor and all its model workers.

The new run restarts the full evaluations; the stopped sequential run's 51
DeltaNet tasks are retained in its original directory and are not mixed into the
new data. Prefill requests only the final-token logits when explicitly supported
by the model, reducing transient memory. All other probe and generation settings
remain the same; control-isolation tests run again with the new configuration.

`latest_parallel.txt` and `latest.txt` point to the concurrent run. Suite status
lists active/completed/failed models; each model has its own status and ETA.
`process.json` stores each worker PID. All five contend for one GPU and shared
memory bandwidth: five workers do not imply fivefold speedup. Initial estimate
is 6–14 hours for all six ranks, to be refined from full-task throughput. The single-worker historic
suite took roughly six hours; concurrency could help idle gaps or add contention.

The all-rank restart uses eight rows at measured steps (native + INT8 + six
SVD ranks). Concurrent real-model smoke checks run again with this larger batch.
Previous stopped runs remain in their original directories. Output method names
are `int8_layer_affine`, `svd1`, `svd2`, `svd4`, `svd8`, `svd16`, and `svd32`.

## Per-head INT8-only rerun

Launch with `bash experiments/all_models_int8_svd_kl/launch_per_head.sh`.
This reuses the prompt file from `latest.txt`, leaves existing results and that
pointer intact, and writes `latest_per_head.txt` for the new durable systemd job.
Five model processes run concurrently. All five eight-token smoke runs must pass
before the full suite starts automatically.

The worker flag `--int8-per-head-only` disables all SVD ranks and SVD groups.
Affine UINT8 quantization retains the earlier 256-level min/max formula, but
reduces only the last two matrix dimensions, using a separate scale and zero-point
for every head in every recurrent layer. The probe batch contains the native
reference and one quantized row. Batch-size-dependent numerical differences may
change greedy reference trajectories relative to the prior eight-row run.

The full protocol retains 100 chat plus 100 code prompts, 256/full prefixes,
128 generated tokens and one discarded probe every four steps: 6,400 measurements
per model and prefix. Outputs include `summary.csv`, `summary.json`, `RESULTS.md`,
and each model's `metrics.jsonl`, `continuations.jsonl`, and `outputs.md`.
`status.json`, `full/<model>/status.json`, and `run.log` record progress.
The initial runtime estimate is 3–6 hours, subject to prompt lengths and GPU load.
