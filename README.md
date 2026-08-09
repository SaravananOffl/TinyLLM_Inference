# TinyLLM 

TinyLLM is a from-scratch implementation of a small decoder-only language model, built to revisit the core components of modern LLMs end to end. Goal is to dive deeper into the inference implementations specifically.

Plan is to cover the full pipeline—from tokenization and Transformer training to autoregressive generation and inference optimization—with an emphasis on understanding the underlying mechanics rather than relying on high-level model libraries.

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
