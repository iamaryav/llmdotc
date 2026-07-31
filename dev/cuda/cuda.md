# Inference 

### Fundamentals
- Thread/block/grid mapping, shared memory tiling, bank conflicts, warp divergence
- Memory model — coalescing, occupancy, latency hiding, roofline analysis
- Numerics — FP32/FP16/BF16/FP8 ranges, error propagation, loss scaling
- Read a model file and know what inference needs (shapes, dtypes, memory budget) before writing a single kernel

---

### Training kernels, forward only
Dataflow order, each step unblocks the next layer in `gpt.py`. These double as inference building blocks (matmul, layernorm, attention forward are shared).

- [ ] encoder_forward.cu
- [ ] layernorm_forward.cu
- [ ] matmul_forward.cu
- [ ] softmax_forward.cu
- [ ] attention_forward.cu
- [ ] trimat_forward.cu
- [ ] crossentropy_forward.cu
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
- [ ] adamw.cu
- [ ] global_norm.cu
- [ ] nccl_all_reduce.cu

**Fused training kernels:**
- [ ] fused_residual_forward.cu
- [ ] classifier_fused.cu
- [ ] permute.cu

---

### Core inference path 
Sequential, each unblocks the next:

- [ ] kv_cache_append.cu — store K/V per token once
- [ ] attention_prefill.cu — process the prompt
- [ ] attention_decode.cu — one new token against cached K/V
- [ ] sampler.cu — GPU-side sampling, no CPU round-trip per token
- [ ] CUDA Graphs / buffer reuse — kill launch + cudaMalloc overhead
- [ ] attention_ops.cu — GQA KV split, sliding window mask, RoPE cache, decode-friendly layout
- [ ] flash_attention.cu — Flash Attention 2/3, tiled IO-aware kernel
- [ ] paged_kv_cache.cu — vLLM-style paging, fixes fragmentation
- [ ] chunked_prefill.cu — split prefill into chunks, interleave with decode so long prompts don't stall other requests' latency (Sarathi's core idea)
- [ ] prefix_cache.cu — reuse KV cache across requests sharing a prompt
- [ ] batched_decode.cu — static batching across sequences
- [ ] continuous_batching.cu — iteration-level scheduler (Orca-style), likely its own executable/API layer
- [ ] quantized_matmul.cu — FP8/int8 weights + activations, PTQ + QAT, touches loading/format too

---

### Know conceptually, implement if time allows
- [ ] speculative_decode.cu — draft/verify with a small model
- [ ] kv_transfer / pd_disaggregation — split prefill and decode across separate GPU pools; understand *why* (prefill is compute-bound, decode is memory-bandwidth-bound — colocating them makes each starve the other) even without hand-rolling the NIXL/RDMA transfer layer
- [ ] tensor_parallel.cu — split weights across GPUs
- [ ] pipeline_parallel.cu — layer sharding for extreme scale
- [ ] stream_scheduler.cu — DAG scheduling, overlap compute across streams
- [ ] gpu_tokenizer.cu — avoid CPU sync per token
- [ ] kv_swap.cu — preemption/swapping when KV cache overflows
- [ ] kv_cache_quant.cu — quantizing the cache itself (INT8/FP8), separate from weight quantization above
- [ ] structured_decoding.cu — grammar/JSON-constrained sampling, extension of sampler.cu

---

### Do alongside everything above, not at the end
- GPU profiling — Nsight Compute (kernel-level), Nsight Systems (timeline), roofline analysis
- Benchmarking discipline — TTFT, TPOT, throughput vs. batch size, warmup + steady-state measurement

---

### Beyond kernels

**Tooling fluency:**
- Debugging — cuda-gdb, compute-sanitizer, deterministic reproduction
- A high-level kernel layer (Triton/TVM/CUTLASS) — write fast, compare against hand-tuned
- Profiling-guided optimization loop — measure first, never guess

**Systems + communication:**
- NCCL collectives, PCIe/NVLink topology, multi-GPU memory management
- Model I/O — safetensors, checkpoint conversion, fused decode preprocessing

**throughout:**
- Serving/product view — latency budgets, TTFT/TPOT tradeoffs, capacity planning
- Maintainable, correctness-first kernels — most real inference code is glue + one hard kernel
- Measure (profiling), compare (baselines), deploy (integration, robustness)
