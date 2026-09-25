## sharegpt_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
 to build a community of advocates and evangel
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0552207 | " build" | " create" |
| 0 | svd16 | 0.000901301 | " build" | " build" |
| 4 | int8_layer_affine | 0.0323728 | " advocates" | " customers" |
| 4 | svd16 | 0.00177852 | " advocates" | " advocates" |

## sharegpt_0000 / prefix -1 (7461 tokens)

Reference continuation:

```text
 of majin majn majn maj
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 1.29339 | " maj" | " the" |
| 0 | svd16 | 0.0422818 | " maj" | " maj" |
| 4 | int8_layer_affine | 6.66563 | " maj" | "," |
| 4 | svd16 | 2.72208 | " maj" | " and" |

## code_thestack_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
self, response):
      print("
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0310461 | "," | "):" |
| 0 | svd16 | 0.0309276 | "," | "):" |
| 4 | int8_layer_affine | 0.00745738 | "      " | "      " |
| 4 | svd16 | 0.000316147 | "      " | "      " |

## code_thestack_0000 / prefix -1 (10074 tokens)

Reference continuation:

```text
<|endoftext|>### Table of character frequencies



```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 9.6838 | "###" | "<\|endoftext\|>" |
| 0 | svd16 | 0.0105195 | "###" | "###" |
| 4 | int8_layer_affine | 0.0111853 | " frequencies" | " frequencies" |
| 4 | svd16 | 0.000325042 | " frequencies" | " frequencies" |

