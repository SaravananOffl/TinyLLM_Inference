# TinyLLM 

TinyLLM is a from-scratch implementation of a small decoder-only language model, built to revisit the core components of modern LLMs end to end. The primary focus is understanding LLM inference from first principles, including autoregressive decoding, KV caching, attention variants, and their latency and memory tradeoffs.

### Tokenization 

The tokenizer is trained from scratch using byte-level BPE with a vocabulary size of 8,192.

To run the training: 
```
python train_tokenizer.py 
```

Example:

Once upon a time

→ ['Once', 'Ġupon', 'Ġa', 'Ġtime']

→ [431, 447, 259, 396]
