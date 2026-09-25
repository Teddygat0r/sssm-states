# Why does full-state INT8 degrade GSM8K?

The completed full-state INT8 run lost 10.24 percentage points against its
baseline, while SVD+INT8 lost 2.20 points. These were different batch sizes and
quantization granularity, so they do not isolate an SVD benefit.

## Paired pilot

`./experiments/qwen35_svd16_int8/launch_int8_diagnostic.sh full`
launches a detached systemd service. `smoke` runs one full-answer question using
the identical launch/code path. `latest_int8_diagnostic_full.txt` points to output.

200 questions are sampled uniformly without replacement from all 1,319 GSM8K
questions using Python random seed 20260922, then sorted by document ID. Selection
uses no outcomes from earlier runs. All six conditions run together in six
independent batch rows using identical cached five-shot prompts, greedy decoding,
768-token cap, and the previously validated GSM8K grader.

| Condition | Scales per head | Compression frequency |
| --- | ---: | --- |
| Native state | — | Never |
| Whole-head INT8 | 1 | Every step |
| Per-row INT8 | 128 | Every step |
| Per-column INT8 | 128 | Every step |
| Whole-head INT8, prefill only | 1 | Once |
| Rank-16 SVD + INT8 factors | 32 | Every step |

All quantization is symmetric signed INT8 [-127,127] with FP32 max-absolute-value
scales. Row/column names refer to the last two state tensor dimensions. All
compressed arms are compressed before the final prompt token; 'every step' also
recompresses before each subsequent decode step. KV/conv states and weights are
untouched. State reconstructions feed dense model kernels. These are quality
experiments, not actual memory/throughput benchmarks.

## Hypotheses and interpretation

1. **Coarse scales:** per-row or per-column INT8 improves accuracy and reduces
   truncations relative to whole-head INT8 at the same frequency. One extreme
   entry otherwise sets the quantization step for all 16,384 entries.
2. **Repeated rounding:** whole-head prefill-only INT8 recovers relative to the
   identical whole-head quantizer applied every step. This tests the frequency
   effect, not just low reconstruction error on the initial state.
3. **Residual low-rank benefit:** if SVD+INT8 still wins after finer scales, the
   retained subspace may matter. This would motivate further tests, not prove
   noise filtering: the representations and storage budgets still differ.

At the shared initial state, save per-arm mean per-head relative squared
reconstruction error, fraction of nonzero entries erased, and mean maxabs/RMS.
These are descriptive mechanism checks, not proxies for generation quality.

Primary endpoint: strict GSM8K accuracy. Secondary: flexible accuracy, number of
answers hitting the cap, generated length, and paired disagreements with native.
Summary includes approximate paired normal 95% intervals for accuracy differences;
200 questions is a diagnostic pilot and cannot resolve very small effects.
Per-row/column quantization uses more scale metadata than factor INT8, so this is
not a matched-memory comparison. All six arms share a batch shape to reduce the
between-run numerical confound noted in the earlier results.

## Outputs

`questions.json`, `config.json`: exact sample, protocol, source hashes.
`samples.jsonl`: incremental generated answers, grades, state metrics.
`summary.json`: running accuracy, truncations, paired differences and intervals.
`status.json`, `run.log`, `unit.txt`: progress, ETA, logs and service name.
`outputs.md`: final readable answers.

Service survives conversation end, not reboot. No automatic retry or resume.
Initial ETA is uncertain because the first question may have atypical length;
status ETA updates after every question. No agent polling is scheduled.
