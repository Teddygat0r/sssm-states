"/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_layer_sweep.py" > kl_layer_sweep.log 2>&1

KL_PARALLEL_WORKERS=8 DATASET=mmlu "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel.log 2>&1

QUANT_BITS=4 "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_quant.py" > quant_4.log 2>&1
QUANT_BITS=8 "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_quant.py" > quant_8.log 2>&1