#### Training (write in order)
Forward pass (dataflow order, each step unblocks the next layer in gpt.py):
- encoder_forward.cu
- layernorm_forward.cu
- matmul_forward.cu
- softmax_forward.cu
- attention_forward.cu
- trimat_forward.cu
- crossentropy_forward.cu
- residual_forward.cu - done
- gelu_forward.cu - done

Backward pass (reverse dataflow order):
- crossentropy_softmax_backward.cu
- attention_backward.cu
- matmul_backward.cu
- matmul_backward_bias.cu
- layernorm_backward.cu
- encoder_backward.cu
- gelu_backward.cu - done

Optimizer / training loop:
- adamw.cu
- global_norm.cu
- nccl_all_reduce.cu

Fused / optimized (write last):
- fused_residual_forward.cu
- classifier_fused.cu
- permute.cu


------------------------------------------------------------------------

#### Inference
Kernels to write, sequentially
- kv_cache_append.cu — store K/V per token once (biggest conceptual win)
- attention_prefill.cu — process the prompt
- attention_decode.cu — one new token against cached K/V
- sampler.cu — GPU-side sampling, no CPU round-trip per token
- CUDA Graphs / buffer reuse — kill launch + cudaMalloc overhead
- attention_ops.cu — GQA KV split, sliding window mask, RoPE cache, decode-friendly MQA layout (permute)
- flash_attention.cu — Flash Attention 2/3, tiled IO-aware kernel
- paged_kv_cache.cu — vLLM-style paging, fixes fragmentation, enables long context
- prefix_cache.cu — reuse KV cache across requests sharing a prompt
- batched_decode.cu — process multiple sequences together (static batching)
- speculative_decode.cu — draft/verify with a small model for 2-3x speedup
- gpu_tokenizer.cu — avoid a CPU sync per token
- kv_swap.cu — preemption & swapping when the KV cache overflows
- quantized_matmul.cu — FP8/int8 with calibration (PTQ + QAT); touches weight format, matmul kernels, accuracy, loading
- continuous_batching.cu — serving scheduler, likely a separate executable / API layer
- tensor_parallel.cu — split weights across GPUs for models larger than one GPU
- pipeline_parallel.cu — layer sharding for extreme scale
- stream_scheduler.cu — DAG scheduling / request multiplexing, overlap compute across streams

Do these alongside, not at the end:
- GPU profiling — Nsight Compute (kernel-level), Nsight Systems (timeline), roofline analysis
- Benchmarking discipline — TTFT, TPOT, throughput vs batch size, warmup + steady-state measurement

------------------------------------------------------------------------

#### Beyond kernels
Coding fundamentals (know by hand, do before the kernel list):
- Thread/block/grid mapping, shared memory tiling, bank conflicts, warp divergence
- Memory model — coalescing, occupancy, latency hiding, roofline analysis
- Numerics — FP32/FP16/BF16/FP8 ranges, error propagation, loss scaling

Tooling fluency:
- Debugging — cuda-gdb, compute-sanitizer, deterministic reproduction
- A high-level kernel layer (Triton / TVM / CUTLASS) — write fast, compare against hand-tuned
- Profiling-guided optimization loop — measure first, never guess

Systems + communication:
- NCCL collectives, PCIe/NVLink topology, multi-GPU memory management
- Model I/O — safetensors, checkpoint conversion, fused decode preprocessing

Engineer mindset (do throughout):
- Read a model file and know what inference needs (shapes, dtypes, memory budget) before writing any kernel
- Serving/product view — latency budgets, TTFT/TPOT tradeoffs, capacity planning
- Maintainable, correctness-first kernels — most real inference code is glue + one hard kernel
- Measure (profiling), compare (baselines), deploy (integration, robustness)

------------------------------------------------------------------------
