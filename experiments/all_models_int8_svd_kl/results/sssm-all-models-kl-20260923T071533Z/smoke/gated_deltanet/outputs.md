## sharegpt_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
to ensure that it is meeting the needs
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00453497 | "ensure" | "ensure" |
| 0 | svd16 | 0.00485063 | "ensure" | "ensure" |
| 4 | int8_layer_affine | 0.0114007 | "meeting" | "meeting" |
| 4 | svd16 | 0.00369413 | "meeting" | "meeting" |

## sharegpt_0000 / prefix -1 (8136 tokens)

Reference continuation:

```text
of the unknown, for the secrets of
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0185944 | "the" | "the" |
| 0 | svd16 | 0.237517 | "the" | "met" |
| 4 | int8_layer_affine | 0.00925903 | "the" | "the" |
| 4 | svd16 | 0.0710512 | "the" | "the" |

## code_thestack_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
get(0))
   def on
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00619582 | "(" | "(" |
| 0 | svd16 | 0.0159497 | "(" | "(" |
| 4 | int8_layer_affine | 0.00730776 | " " | " " |
| 4 | svd16 | 0.0272555 | " " | " " |

## code_thestack_0000 / prefix -1 (11395 tokens)

Reference continuation:

```text
decoding_table = encoding_table
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00162227 | "oding" | "oding" |
| 0 | svd16 | 0.0570408 | "oding" | "oding" |
| 4 | int8_layer_affine | 0.0112884 | "encoding" | "code" |
| 4 | svd16 | 0.206042 | "encoding" | "encoding" |

