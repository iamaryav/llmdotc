# Plan
- write arch in torch - GPT - 2/3 size
- write decoder only arch then improve one by one
- shakespeare/tiny stories or tiny general data
- write training pipeline in torch
- do a training run(verify the correctness and loss graph)
- write in c/cuda(piece by piece) 
- speed comparision with pytorch training loop
- Benchmark comparision
- explain what i learned posts/Share/ask/explain/blogs throughout the project
#### GPT architecture
- [GPTConfig]()
- [Transformer decoder only architecture](https://arxiv.org/pdf/1706.03762v1) - Done
- [GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - wip
- [RoPE - Rotary Positional Embedding](https://arxiv.org/abs/2104.09864) - Done
- PreNorm (https://arxiv.org/abs/2002.04745) - Done
- [RMSNorm - No learnable params](https://arxiv.org/abs/1910.07467)
- No bias in linear layers - Done
- [QK Norm](https://arxiv.org/abs/2010.04245) - Done
- [Group Query Attention - GQA](https://arxiv.org/abs/2406.14963) - Done
- Attention dropout - Not in this arch
- [Untied weights for token embeddings and lm_head]() - Done

- Generate method - Done
- KV-Cache implementation - while doing inference - Done
- Activation Funtions: ReLU, GeLU, SwigLU, ReLU ** 2
- [relu**2 activation in MLP](https://arxiv.org/abs/2402.03804)
- [Sliding window with Causal](https://arxiv.org/abs/2502.18845)

- Optimizer setup [AdamW](https://arxiv.org/abs/1711.05101)
- [Muon Optimizer](https://arxiv.org/abs/2502.16982)
- [initialize model method]() - next
- [Xavier/Glorot initialization](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) for linear layers
- [He/Kaiming initialization](https://arxiv.org/abs/1502.01852) for ReLU activations
- estimate flops method
- method scaling law - [Kaplan et al.](https://arxiv.org/abs/2001.08361) and [Chinchilla](https://arxiv.org/abs/2203.15556) optimal compute and parameter counts

- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Flash Attention-2](https://arxiv.org/abs/2307.08691)
- [Flash Attention-3](https://arxiv.org/abs/2407.08608)
- Inference with flash attention
- for now will use existing transformers

#### Training Pipeline
- [DDP training pipeline](https://arxiv.org/abs/2006.15704)
- [FSDP](https://arxiv.org/abs/2304.11280)(later if needed)
- Tokenizer
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
- Evaluation/validation loop/visualization
- Wandb integration
