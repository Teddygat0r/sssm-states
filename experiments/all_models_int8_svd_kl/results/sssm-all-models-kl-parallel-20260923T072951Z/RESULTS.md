# Five-model INT8 and SVD: complete aggregate results

Completed September 23, 2026. All five models passed smoke checks and completed 400 prompt-prefix tasks each. The report includes all 70 aggregate groups (448,000 probe measurements).

## Protocol

- 100 ShareGPT and 100 code prompts, each evaluated at a 256-token prefix and full prompt (65,536-token safety cap). Code uses the repository’s CodeParrot fallback.
- 128-token greedy reference continuations; probes every four steps. Each model/method/prefix group has 6,400 measurements.
- INT8 uses affine min/max quantization with one scale and zero point per recurrent layer, shared across heads. SVD uses ranks 1, 2, 4, 8, 16, 32 without factor quantization.
- SVD ranks 4/8/16 share a rank-16 sketch with oversampling 8; ranks 1/2/32 share a rank-32 sketch with oversampling 0. Both use two power iterations, matching the historical sweep settings.
- All methods probe identical reference states. All recurrent heads/layers are compressed together; attention KV and convolution states stay unchanged. Probe caches are discarded, so these results measure single-step sensitivity, not accumulated rollout drift.
- KL is D_KL(reference || probe), in nats. Percentiles use nearest rank. Top-1 agreement compares next-token argmax predictions.
- Reconstruction error is mean per-head squared relative Frobenius error, ||S − S_hat||_F² / ||S||_F², averaged over probes. Methods are not matched for storage cost.

[Machine-readable CSV](summary.csv) · [JSON](summary.json) · [Suite configuration](config.json)

## Qwen3.5-4B

[Reference continuations and probe predictions](full/qwen35/outputs.md) · [Raw probe records](full/qwen35/metrics.jsonl)

### 256-token prefix

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.11992655 | 0.01584838 | 0.21877114 | 1.8771793 | 14.48897 | 91.7969% | 0.091581168 |
| SVD rank 1 | 6400 | 0.11675325 | 0.014899112 | 0.21823832 | 1.7485633 | 14.654137 | 91.6406% | 0.19074449 |
| SVD rank 2 | 6400 | 0.064520999 | 0.0064126691 | 0.099663578 | 0.92714065 | 15.640671 | 94.2344% | 0.10835718 |
| SVD rank 4 | 6400 | 0.036754352 | 0.0023337859 | 0.037212394 | 0.37722796 | 16.203262 | 96.3750% | 0.054611682 |
| SVD rank 8 | 6400 | 0.020898296 | 0.00081663218 | 0.013030408 | 0.12865314 | 13.721222 | 98.0312% | 0.022896976 |
| SVD rank 16 | 6400 | 0.0077746173 | 0.00025262145 | 0.0043546185 | 0.035369989 | 9.0197792 | 98.7344% | 0.0073045212 |
| SVD rank 32 | 6400 | 0.0034541615 | 7.5177901e-05 | 0.001877698 | 0.0077479742 | 3.6951284 | 99.2344% | 0.0015718232 |

### Full prompt

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.1171295 | 0.016797677 | 0.23544452 | 1.6674627 | 16.248806 | 92.6562% | 0.09832096 |
| SVD rank 1 | 6400 | 0.09835264 | 0.010669567 | 0.18790051 | 1.2850342 | 12.500666 | 93.6406% | 0.19987221 |
| SVD rank 2 | 6400 | 0.050079019 | 0.0048645679 | 0.089199595 | 0.53716594 | 11.543626 | 95.5938% | 0.11762405 |
| SVD rank 4 | 6400 | 0.028171749 | 0.0017564722 | 0.041324664 | 0.2751368 | 10.601533 | 97.1406% | 0.062599703 |
| SVD rank 8 | 6400 | 0.01506048 | 0.0005057971 | 0.016307388 | 0.10310078 | 10.705208 | 98.1875% | 0.028729297 |
| SVD rank 16 | 6400 | 0.0073171522 | 0.00016278531 | 0.006382098 | 0.046020381 | 5.6350532 | 98.6406% | 0.010613841 |
| SVD rank 32 | 6400 | 0.002670924 | 5.5944904e-05 | 0.0023975167 | 0.014235029 | 4.4129539 | 99.0000% | 0.0029149317 |

## Nemotron-3-Nano-4B

[Reference continuations and probe predictions](full/nemotron/outputs.md) · [Raw probe records](full/nemotron/metrics.jsonl)

### 256-token prefix

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.8947925 | 0.21790354 | 2.6152296 | 8.3141642 | 17.376629 | 71.2812% | 0.59122888 |
| SVD rank 1 | 6400 | 0.11754881 | 0.010218158 | 0.21630198 | 2.1876347 | 12.357306 | 91.3438% | 0.039668754 |
| SVD rank 2 | 6400 | 0.052796859 | 0.0035524222 | 0.074148327 | 0.89992374 | 7.539196 | 94.8750% | 0.015125858 |
| SVD rank 4 | 6400 | 0.016515097 | 0.0011069601 | 0.023340361 | 0.20941366 | 9.0914679 | 97.2969% | 0.0055198726 |
| SVD rank 8 | 6400 | 0.0042947502 | 0.00028681813 | 0.0054894718 | 0.032593988 | 8.6188097 | 98.6406% | 0.0017496924 |
| SVD rank 16 | 6400 | 0.00075129649 | 7.9346668e-05 | 0.0019274289 | 0.0064036101 | 0.11319822 | 99.3125% | 0.00043942214 |
| SVD rank 32 | 6400 | 0.00037291812 | 2.9866555e-05 | 0.0013694932 | 0.0027649328 | 0.019434258 | 99.4219% | 7.2844033e-05 |

### Full prompt

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.54140168 | 0.026480652 | 1.4740746 | 7.6710825 | 19.036871 | 85.4688% | 0.57768856 |
| SVD rank 1 | 6400 | 0.070620339 | 0.010599911 | 0.10728084 | 1.3244705 | 14.461682 | 96.2969% | 0.032451569 |
| SVD rank 2 | 6400 | 0.021872079 | 0.0018916711 | 0.02702551 | 0.38090953 | 2.8954301 | 97.8750% | 0.012459197 |
| SVD rank 4 | 6400 | 0.0070991782 | 0.00013821351 | 0.0092130899 | 0.10707058 | 2.2816308 | 98.7188% | 0.0047179529 |
| SVD rank 8 | 6400 | 0.0017894336 | 4.4577264e-05 | 0.0027959235 | 0.028165065 | 0.80150712 | 99.3750% | 0.0016209047 |
| SVD rank 16 | 6400 | 0.00044969064 | 1.3917966e-05 | 0.0012197811 | 0.0063904449 | 0.14379388 | 99.6094% | 0.00046063525 |
| SVD rank 32 | 6400 | 0.0001996724 | 4.3848031e-06 | 0.00056777737 | 0.002158165 | 0.01756731 | 99.7969% | 9.0926908e-05 |

## Mamba2-1.3B

[Reference continuations and probe predictions](full/mamba2/outputs.md) · [Raw probe records](full/mamba2/metrics.jsonl)

### 256-token prefix

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.42903498 | 0.022620194 | 0.73727667 | 8.2334719 | 26.875143 | 87.3125% | 0.27001339 |
| SVD rank 1 | 6400 | 0.66724076 | 0.15256158 | 1.9821188 | 6.0752993 | 12.217052 | 75.1562% | 0.13855522 |
| SVD rank 2 | 6400 | 0.39562565 | 0.067139521 | 1.1353998 | 4.2012691 | 10.030181 | 83.2500% | 0.068683557 |
| SVD rank 4 | 6400 | 0.17949812 | 0.019170105 | 0.43263161 | 2.7483554 | 9.8383665 | 90.8281% | 0.029097269 |
| SVD rank 8 | 6400 | 0.041619626 | 0.0037683444 | 0.062606186 | 0.79226744 | 8.5494289 | 96.1875% | 0.010440884 |
| SVD rank 16 | 6400 | 0.0050568834 | 0.00068193232 | 0.0085860118 | 0.067934006 | 1.8451221 | 98.3906% | 0.0028383093 |
| SVD rank 32 | 6400 | 0.0012213374 | 0.00018708268 | 0.0025654826 | 0.012750067 | 0.7089752 | 99.2188% | 0.00042721991 |

### Full prompt

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 1.0052102 | 0.46654582 | 2.6401849 | 7.5172596 | 22.22629 | 69.6719% | 0.32735781 |
| SVD rank 1 | 6400 | 0.66584935 | 0.27236989 | 1.8444637 | 4.5880075 | 13.380378 | 75.6250% | 0.08720697 |
| SVD rank 2 | 6400 | 0.32297344 | 0.08598537 | 0.92556989 | 2.9485621 | 8.8296328 | 85.7344% | 0.040612914 |
| SVD rank 4 | 6400 | 0.15075198 | 0.028148862 | 0.38065192 | 1.8664068 | 8.0508804 | 90.9531% | 0.017453126 |
| SVD rank 8 | 6400 | 0.055128301 | 0.0088314777 | 0.11800861 | 0.80509377 | 4.2394481 | 94.4844% | 0.0069837527 |
| SVD rank 16 | 6400 | 0.017862854 | 0.0034620883 | 0.033353224 | 0.26387581 | 3.7287288 | 96.8906% | 0.0023419962 |
| SVD rank 32 | 6400 | 0.0056114786 | 0.0013848445 | 0.010969363 | 0.068843126 | 0.55427009 | 98.1250% | 0.00047997134 |

## DeltaNet-1.3B

[Reference continuations and probe predictions](full/deltanet/outputs.md) · [Raw probe records](full/deltanet/metrics.jsonl)

### 256-token prefix

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.36664246 | 0.097678274 | 1.1153998 | 3.3049455 | 7.4056315 | 84.8594% | 0.021895264 |
| SVD rank 1 | 6400 | 2.6845072 | 1.9607685 | 6.256485 | 10.094883 | 14.698743 | 38.8906% | 0.22572245 |
| SVD rank 2 | 6400 | 1.8300927 | 1.1518935 | 4.5342455 | 8.7179346 | 13.026589 | 49.5625% | 0.11640363 |
| SVD rank 4 | 6400 | 0.83479703 | 0.37272716 | 2.336349 | 5.2752614 | 10.363287 | 69.6562% | 0.051007745 |
| SVD rank 8 | 6400 | 0.30544513 | 0.079372816 | 0.87857461 | 2.8758011 | 9.7632675 | 86.1562% | 0.021312059 |
| SVD rank 16 | 6400 | 0.083080513 | 0.010896489 | 0.1807051 | 1.164614 | 5.9060888 | 95.2031% | 0.0084996727 |
| SVD rank 32 | 6400 | 0.0056191137 | 0.00085844984 | 0.010126716 | 0.079964519 | 0.81206805 | 98.8750% | 0.0024745517 |

### Full prompt

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.079167453 | 0.014695406 | 0.19168845 | 0.9820134 | 4.7810655 | 94.1406% | 0.031972395 |
| SVD rank 1 | 6400 | 2.6606007 | 1.8942852 | 6.093081 | 10.671746 | 23.263868 | 40.5469% | 0.4822848 |
| SVD rank 2 | 6400 | 1.8990499 | 1.2654015 | 4.6396704 | 8.4192419 | 13.695172 | 50.9219% | 0.33232249 |
| SVD rank 4 | 6400 | 1.2478263 | 0.67903703 | 3.2161515 | 6.7731152 | 13.30049 | 61.8438% | 0.22671704 |
| SVD rank 8 | 6400 | 0.70719335 | 0.30649272 | 1.9345392 | 4.7891979 | 15.317448 | 74.5156% | 0.15125406 |
| SVD rank 16 | 6400 | 0.30304343 | 0.091986284 | 0.79899621 | 2.8562248 | 17.917292 | 86.2969% | 0.090835274 |
| SVD rank 32 | 6400 | 0.076319419 | 0.01560186 | 0.17738026 | 0.84070647 | 14.163464 | 94.1250% | 0.042392232 |

## GatedDeltaNet-1.3B

[Reference continuations and probe predictions](full/gated_deltanet/outputs.md) · [Raw probe records](full/gated_deltanet/metrics.jsonl)

### 256-token prefix

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.0034155691 | 0.0008819116 | 0.0090356916 | 0.032383405 | 0.25546548 | 98.4531% | 0.075683381 |
| SVD rank 1 | 6400 | 1.8442814 | 0.82675254 | 5.2073393 | 9.7102966 | 15.076679 | 52.8281% | 0.4128434 |
| SVD rank 2 | 6400 | 1.287525 | 0.48429316 | 3.821039 | 7.4952893 | 15.107764 | 61.5156% | 0.2800955 |
| SVD rank 4 | 6400 | 0.74193485 | 0.22080176 | 2.2119751 | 5.5937452 | 11.753845 | 74.9219% | 0.17574761 |
| SVD rank 8 | 6400 | 0.29220733 | 0.060532402 | 0.78618628 | 3.1114609 | 8.8037796 | 87.7344% | 0.098991377 |
| SVD rank 16 | 6400 | 0.072604302 | 0.010338617 | 0.15943772 | 1.0717621 | 5.0982208 | 95.0000% | 0.048021381 |
| SVD rank 32 | 6400 | 0.013898842 | 0.0019052927 | 0.025082007 | 0.20328718 | 1.9340998 | 97.8125% | 0.018425025 |

### Full prompt

| Method | N | Mean KL | Median KL | p90 KL | p99 KL | Max KL | Top-1 agreement | Mean squared relative Frobenius error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 (per layer) | 6400 | 0.0049145052 | 0.0011460743 | 0.013426363 | 0.044618949 | 0.55056 | 98.4531% | 0.085898324 |
| SVD rank 1 | 6400 | 2.3216672 | 1.2330149 | 6.3350539 | 11.135531 | 17.520819 | 47.8281% | 0.42572528 |
| SVD rank 2 | 6400 | 1.8842497 | 0.95801163 | 5.2111835 | 9.5462427 | 20.981836 | 53.6563% | 0.29862267 |
| SVD rank 4 | 6400 | 1.3223063 | 0.55309838 | 3.7947624 | 7.7684417 | 15.57999 | 63.0156% | 0.19873598 |
| SVD rank 8 | 6400 | 0.67908158 | 0.2043741 | 1.9106585 | 5.4290891 | 11.327115 | 77.0156% | 0.12480789 |
| SVD rank 16 | 6400 | 0.2588906 | 0.051443938 | 0.69180697 | 3.053112 | 10.114284 | 88.3594% | 0.072852716 |
| SVD rank 32 | 6400 | 0.079690151 | 0.012012232 | 0.18122858 | 1.1722419 | 7.985857 | 93.7656% | 0.037391054 |

