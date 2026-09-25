## sharegpt_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
to ensure that the product is performing as
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.032092 | "ensure" | "ensure" |
| 0 | svd16 | 0.00552275 | "ensure" | "ensure" |
| 4 | int8_layer_affine | 0.16965 | "is" | "is" |
| 4 | svd16 | 0.0128328 | "is" | "is" |

## sharegpt_0000 / prefix -1 (8136 tokens)

Reference continuation:

```text
of machine language programming!

====
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.319549 | "machine" | "the" |
| 0 | svd16 | 0.520874 | "machine" | "the" |
| 4 | int8_layer_affine | 0.00401392 | "\n" | "\n" |
| 4 | svd16 | 0.00512968 | "\n" | "\n" |

## code_thestack_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
status))
      self.send(
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.170885 | "))" | "_" |
| 0 | svd16 | 0.00965965 | "))" | "))" |
| 4 | int8_layer_affine | 0.00494017 | "." | "." |
| 4 | svd16 | 0.00156883 | "." | "." |

## code_thestack_0000 / prefix -1 (11395 tokens)

Reference continuation:

```text
encoding_table=codecs.register
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00272868 | "_" | "_" |
| 0 | svd16 | 0.0522526 | "_" | "_" |
| 4 | int8_layer_affine | 4.10169e-06 | "cs" | "cs" |
| 4 | svd16 | 0.00239573 | "cs" | "cs" |

