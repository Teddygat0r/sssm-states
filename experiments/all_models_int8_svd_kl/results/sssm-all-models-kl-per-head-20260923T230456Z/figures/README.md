# Rank vs KL plots

SVD source: /home/joshuaz/sssm-states/experiments/all_models_int8_svd_kl/results/sssm-all-models-kl-parallel-20260923T072951Z

Per-head INT8 source: /home/joshuaz/sssm-states/experiments/all_models_int8_svd_kl/results/sssm-all-models-kl-per-head-20260923T230456Z

Five model panels: three above two centered below. Blue: 256-token prefix; orange: full prompt. Y axis is logarithmic; x axis shows SVD ranks. Per-head INT8 is shown as horizontal reference lines; per-layer INT8 is omitted. No experiments were rerun. The combined figure uses solid lines / filled markers for mean and dotted lines / hollow markers for p99. The mean and p99 figures use the corresponding saved aggregate statistics.
