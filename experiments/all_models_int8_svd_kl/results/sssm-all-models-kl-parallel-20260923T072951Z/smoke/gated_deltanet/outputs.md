## sharegpt_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
to ensure that it is meeting the needs
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00453497 | "ensure" | "ensure" |
| 0 | svd1 | 0.255923 | "ensure" | "ensure" |
| 0 | svd2 | 0.0593605 | "ensure" | "ensure" |
| 0 | svd4 | 0.0134079 | "ensure" | "ensure" |
| 0 | svd8 | 0.0115046 | "ensure" | "ensure" |
| 0 | svd16 | 0.00485063 | "ensure" | "ensure" |
| 0 | svd32 | 0.00232269 | "ensure" | "ensure" |
| 4 | int8_layer_affine | 0.0114007 | "meeting" | "meeting" |
| 4 | svd1 | 0.22514 | "meeting" | "meeting" |
| 4 | svd2 | 0.113935 | "meeting" | "meeting" |
| 4 | svd4 | 0.0266743 | "meeting" | "meeting" |
| 4 | svd8 | 0.00554947 | "meeting" | "meeting" |
| 4 | svd16 | 0.00405727 | "meeting" | "meeting" |
| 4 | svd32 | 0.000884298 | "meeting" | "meeting" |

## sharegpt_0000 / prefix -1 (8136 tokens)

Reference continuation:

```text
of the unknown, for the secrets of
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.0185944 | "the" | "the" |
| 0 | svd1 | 1.22483 | "the" | "the" |
| 0 | svd2 | 1.12341 | "the" | "the" |
| 0 | svd4 | 0.810059 | "the" | "the" |
| 0 | svd8 | 0.626407 | "the" | "met" |
| 0 | svd16 | 0.237517 | "the" | "met" |
| 0 | svd32 | 0.0928779 | "the" | "the" |
| 4 | int8_layer_affine | 0.00925903 | "the" | "the" |
| 4 | svd1 | 0.165093 | "the" | "the" |
| 4 | svd2 | 0.107005 | "the" | "there" |
| 4 | svd4 | 0.160101 | "the" | "there" |
| 4 | svd8 | 0.14085 | "the" | "there" |
| 4 | svd16 | 0.0641506 | "the" | "the" |
| 4 | svd32 | 0.029998 | "the" | "the" |

## code_thestack_0000 / prefix 256 (256 tokens)

Reference continuation:

```text
get(0))
   def on
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00619582 | "(" | "(" |
| 0 | svd1 | 0.197464 | "(" | "(" |
| 0 | svd2 | 0.1644 | "(" | "(" |
| 0 | svd4 | 0.0574154 | "(" | "(" |
| 0 | svd8 | 0.0290199 | "(" | "(" |
| 0 | svd16 | 0.0159497 | "(" | "(" |
| 0 | svd32 | 0.00274174 | "(" | "(" |
| 4 | int8_layer_affine | 0.00730776 | " " | " " |
| 4 | svd1 | 4.573 | " " | "The" |
| 4 | svd2 | 1.83196 | " " | "This" |
| 4 | svd4 | 0.54515 | " " | " " |
| 4 | svd8 | 0.0710185 | " " | " " |
| 4 | svd16 | 0.0270383 | " " | " " |
| 4 | svd32 | 0.00562882 | " " | " " |

## code_thestack_0000 / prefix -1 (11395 tokens)

Reference continuation:

```text
decoding_table = encoding_table
```

| Step | Method | KL | Reference next token | Probe next token |
|---|---|---:|---|---|
| 0 | int8_layer_affine | 0.00162227 | "oding" | "oding" |
| 0 | svd1 | 1.26308 | "oding" | "oding" |
| 0 | svd2 | 0.816887 | "oding" | "oding" |
| 0 | svd4 | 0.43288 | "oding" | "oding" |
| 0 | svd8 | 0.199829 | "oding" | "oding" |
| 0 | svd16 | 0.0570408 | "oding" | "oding" |
| 0 | svd32 | 0.0112652 | "oding" | "oding" |
| 4 | int8_layer_affine | 0.0112884 | "encoding" | "code" |
| 4 | svd1 | 2.31991 | "encoding" | "\"" |
| 4 | svd2 | 1.64969 | "encoding" | "\"" |
| 4 | svd4 | 1.28076 | "encoding" | "decode" |
| 4 | svd8 | 0.320811 | "encoding" | "decode" |
| 4 | svd16 | 0.20357 | "encoding" | "encoding" |
| 4 | svd32 | 0.0673371 | "encoding" | "encoding" |

