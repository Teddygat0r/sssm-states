# "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_layer_sweep.py" > kl_layer_sweep.log 2>&1

echo "Running KL parallel with rank 1"
KL_PARALLEL_WORKERS=8 RANK="1" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_1.log 2>&1
echo "Running KL parallel with rank 2"
KL_PARALLEL_WORKERS=8 RANK="2" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_2.log 2>&1
echo "Running KL parallel with rank 4"
KL_PARALLEL_WORKERS=8 RANK="4" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_4.log 2>&1
echo "Running KL parallel with rank 6"
KL_PARALLEL_WORKERS=8 RANK="6" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_6.log 2>&1
echo "Running KL parallel with rank 8"
KL_PARALLEL_WORKERS=8 RANK="8" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_8.log 2>&1
echo "Running KL parallel with rank 10"
KL_PARALLEL_WORKERS=8 RANK="10" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_10.log 2>&1
echo "Running KL parallel with rank 12"
KL_PARALLEL_WORKERS=8 RANK="12" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_12.log 2>&1
echo "Running KL parallel with rank 16"
KL_PARALLEL_WORKERS=8 RANK="16" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_16.log 2>&1
echo "Running KL parallel with rank 24"
KL_PARALLEL_WORKERS=8 RANK="24" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_24.log 2>&1
echo "Running KL parallel with rank 32"
KL_PARALLEL_WORKERS=8 RANK="32" "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_kl_parallel.py" > kl_parallel_32.log 2>&1

# QUANT_BITS=4 "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_quant.py" > quant_4.log 2>&1
# QUANT_BITS=8 "/home/joshuaz/sssm-states/.venv/bin/python" "state_spectrum_sweep/run_experiment_quant.py" > quant_8.log 2>&1