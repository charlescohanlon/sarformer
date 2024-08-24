# DOCUMENTATION

## Random Notes



## Tokenization

### Metadata (Strutured Data) [NEEDS TO BE UPDATED]

Building off of 4M's approach, which uses several vX=Y (0<=X<=4, 0<=Y<=999) tokens to communicate the value of metadata, we're using v0=(something) and v1=(hexadecimal of float32 value). Where v0 denotes the type of metadata, and v1 indicates the value of said metadata attribute. v0 ranges from [0, # types of metadata].


Example of three metadata representations for 3 types of metadata:
```
v0=2 v1=A8210D4A v0=0 v1=AFFEB3A2 v0=1 v1=E34B0C9F
```

Order of a pair of v0 and v1 doesn't matter (and is randomized), but a v0 is always followed by a v1 that denotes it's value.

We're using this approach because it allows us to represent arbitrarily large floating point values without a large vocabulary size and without increasing the sequence length dramatically. It seems 4M used values in the range [0, 999], which gives their metadata attributes 1000 possible values. They included a different token in the vocabulary for each possible value (see fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json). This route is unsatisfactory if we want the ability to represent arbitrary floating point values (we'd need a vocab size of over 10^38). Another potential route is to only have tokens for individual digits (base 10) but then sequence length would increase with number precision (and magnitude) very quickly.

Our solution is to use hex to represent the IEEE 754 standard for the floating point numbers. 32-bit floating point fixes each value to a sequence length of 4 bytes (8 hex digits). We then provide special tokens for every combination of 2 hex digits (i.e., one byte). So a 32-bit floating point value is represented by only 4 tokens, and the vocab size only increased by 256.

It's worth noting that varying the number system base used to represent values will shift the sequence length and token counts. E.g., base 2 => longer sequence and fewer tokens, base 36 => shorter sequence and more tokens.

This idea is kind of similar to byte-level BPE.