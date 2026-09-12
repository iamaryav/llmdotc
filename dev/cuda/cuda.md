# Inference

### Fundamentals

- Thread/block/grid mapping, shared memory tiling, bank conflicts, warp divergence
- Memory model — coalescing, occupancy, latency hiding, roofline analysis
- Numerics — FP32/FP16/BF16/FP8 ranges, error propagation, loss scaling
- Read a model file and know what inference needs (shapes, dtypes, memory budget) before writing a single kernel

---

### Training kernels, forward only

Dataflow order, each step unblocks the next layer in `gpt.py`. These double as inference building blocks (matmul, layernorm, attention forward are shared).

residual_forward.cu -> gelu_forward.cu -> gelu_backward.cu -> adamw.cu -> crossentropy_forward.cu -> crossentropy_softmax_backward.cu -> encoder_forward.cu -> encoder_backward.cu -> softmax_forward.cu -> layernorm_forward.cu -> layernorm_backward.cu -> matmul_backward_bias.cu -> matmul_forward.cu -> trimat_forward.cu -> matmul_backward.cu -> fused_residual_forward.cu -> classifier_fused.cu -> global_norm.cu -> attention_forward.cu -> sampler.cu -> kv_cache_append.cu -> attention_prefill.cu -> attention_decode.cu -> attention_backward.cu -> permute.cu -> nccl_all_reduce.cu

- [ ] encoder_forward.cu
- [ ] layernorm_forward.cu
- [ ] matmul_forward.cu
- [ ] softmax_forward.cu
- [ ] attention_forward.cu
- [ ] trimat_forward.cu
- [x] crossentropy_forward.cu
- [x] residual_forward.cu
- [x] gelu_forward.cu



### Lower priority / optional (training-specific, not inference-relevant)

**Backward pass** (reverse dataflow order):

- [ ] crossentropy_softmax_backward.cu
- [ ] attention_backward.cu
- [ ] matmul_backward.cu
- [ ] matmul_backward_bias.cu
- [ ] layernorm_backward.cu
- [ ] encoder_backward.cu
- [x] gelu_backward.cu

**Optimizer / distributed:**

- [x] adamw.cu
- [ ] global_norm.cu
- [ ] nccl_all_reduce.cu

**Fused training kernels:**

- [ ] fused_residual_forward.cu
- [ ] classifier_fused.cu
- [ ] permute.cu

---

#### Core inference path
Play with the models and their weights
Inference Engineering Book
- Quantization
- Speculative Decoding
- caching (prefill + decoding)
- Flash attention, Paged attention, chunked prefill
- Model parallelism
- Dissaggregation
- Production