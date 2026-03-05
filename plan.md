# Plan

#### GPT architecture
- [GPTConfig]()
- [initialize model method]()
- [Transformer decoder only architecture](https://arxiv.org/pdf/1706.03762v1)
- [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Norm after token embedding
- [RoPE - Rotary Positional Embedding](https://arxiv.org/abs/2104.09864)
- PreNorm (https://arxiv.org/abs/2002.04745)
- [RMSNorm - No learnable params](https://arxiv.org/abs/1910.07467)
- No bias in linear layers
- [QK Norm](https://arxiv.org/abs/2010.04245)
- [Group Query Attention - GQA](https://arxiv.org/abs/2406.14963)
- KV-Cache implementation
- Attention dropout
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Flash Attention-2](https://arxiv.org/abs/2307.08691)
- [Flash Attention-3](https://arxiv.org/abs/2407.08608)
- Inference with flash attention
- [relu**2 activation in MLP](https://arxiv.org/abs/2402.03804)
- [SwiGLU FFN](https://arxiv.org/pdf/2002.05202) # will not use this one
- [Untied weights for token embeddings and lm_head]()
- [Sliding window with Causal](https://arxiv.org/abs/2502.18845)
- for now will use existing transformers
- Generate method
- estimate flops method
- method scaling law - [Kaplan et al.](https://arxiv.org/abs/2001.08361) and [Chinchilla](https://arxiv.org/abs/2203.15556) optimal compute and parameter counts
- Optimizer setup [AdamW](https://arxiv.org/abs/1711.05101)
- [Muon Optimizer](https://arxiv.org/abs/2502.16982)

#### Training Pipeline
- [DDP training pipeline](https://arxiv.org/abs/2006.15704)
- [FSDP](https://arxiv.org/abs/2304.11280)
- data / token generation
- Data loading
- [LR scheduler](https://arxiv.org/abs/2406.09405)
- Gradient accumulation
- [Mixed precision training](https://arxiv.org/abs/1710.03740)
- Training loop - forward, backward, optimizer step
- [Learning rate warmup](https://arxiv.org/abs/2406.09405)
- training log and other evaluation logging, performance profiling
- Checkpointing/saving
- resume training from checkpoint
- [gradient clipping](https://arxiv.org/abs/1905.11881)
- Sampling
- Evaluation/validation loop
- Wandb integration