## sharegpt_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
 are engaged and loyal to the brand.
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0514046 | " engaged" | " engaged" |
| 0 | svd16 | 0.000980576 | " engaged" | " engaged" |
| 4 | int8_layer_affine | 0.000543688 | " the" | " the" |
| 4 | svd16 | 0.00254823 | " the" | " the" |

## sharegpt_0000 / prefix -1 (7062 tokens)

Reference continuation:

```text
 of the unknown, where the mysteries of
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.167447 | " the" | " machine" |
| 0 | svd16 | 0.0128417 | " the" | " the" |
| 4 | int8_layer_affine | 0.0463298 | " the" | " the" |
| 4 | svd16 | 0.00201566 | " the" | " the" |

## code_thestack_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
         yield from asyncio.sleep(1)
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0195203 | " yield" | " yield" |
| 0 | svd16 | 0.00151703 | " yield" | " yield" |
| 4 | int8_layer_affine | 0.0149673 | "(" | "(" |
| 4 | svd16 | 0.000129061 | "(" | "(" |

## code_thestack_0000 / prefix -1 (9576 tokens)

Reference continuation:

```text
# ----- # 

"""
The :
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0382555 | " -----" | " -----" |
| 0 | svd16 | 0.015582 | " -----" | " -----" |
| 4 | int8_layer_affine | 0.013604 | "\n" | "\n" |
| 4 | svd16 | 0.00570483 | "\n" | "\n" |

