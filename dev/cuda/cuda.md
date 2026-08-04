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
Download a hardware-matched target model and apply these techniques to it. Target: `Qwen/Qwen3-0.6B` — 1.5 GB BF16 weights (1.19 GB after dedup'ing the tied `lm_head`), GQA + RoPE + RMSNorm + SwiGLU + QK-norm, the same dense stack as our training pipeline, and it fits the GTX 1650's 4 GB with room for KV cache + batching. Baseline for comparison: llama.cpp GGUF (Q8_0/Q4_K_M) of the same model. Everything here is size-agnostic — prefill/decode split, paged KV, chunked prefill, quantization all demonstrate identically at 0.6B and prove out on real metrics (TTFT, TPOT, bandwidth utilization).

### Phase 0 — before the first kernel
- [ ] Memory budget + roofline — from config.json: weights 1.19 GB (BF16, lm_head dedup'ed), KV cache 112 KB/token (28 × 8 kv-heads × 128 head-dim × 2 × 2B) → 4K ctx 448 MB, 8K ctx 896 MB; working set at 8K ≈ 2.4 GB (32K doesn't fit — why context_mgmt.cu exists). Decode ceiling on ~128 GB/s ≈ 107 tok/s; this is the benchmark anchor.
- [ ] tools/convert_hf_to_bin.py — one-time Python converter: config.json → model_config (header + binary), tokenizer.json → tokenizer.bin (BPE vocab + merge ranks + special tokens + pre-tokenizer rules), safetensors → weights.bin (flat index: name/offset/dtype/shape; tied lm_head dedup'ed)
- [ ] tools/convert.c — C reimplementation of the same converter (hand-rolled JSON reader); output must byte-match the Python version
- [ ] torch/python code to load the model and do inference first
- [ ] for all the cu files implment torch/python as well for faster implementation
- [ ] load_weights.cu — read weights.bin index, upload BF16 to GPU, in-kernel BF16→FP32 dequant (lossless; storage stays BF16, compute runs FP32 — sm_75 has no tensor cores, so FP16/BF16 math buys nothing)
- [ ] tokenizer.cu — read tokenizer.bin: byte-level BPE + Qwen pre-tokenizer + special tokens + Instruct chat template; must be correct before any token is generated
- [ ] sampler.cu — GPU-side sampling: top-k/top-p/min-p filtering, temperature, repetition penalty, EOS/stop handling, beam search; no CPU round-trip per token
- [ ] tokenizer_parity.cu — byte-exact token ID arrays vs `transformers.AutoTokenizer.encode()` on a fixed string set (first correctness gate)
- [ ] model_forward.cu — assemble the shared kernels into the config-driven decoder block (RMSNorm → GQA attention w/ RoPE+QK-norm → SwiGLU MLP → residual), loop all 28 layers, wire weights from load_weights.cu, tie embed_tokens/lm_head. This is the model itself — naive_generate drives it, and prefill/decode reuse the same block later. Qwen needs the RMSNorm + SiLU/SwiGLU variants, distinct from the GPT's LayerNorm + GELU.
- [ ] logits_parity.cu — forward parity vs PyTorch Qwen3-0.6B on the same prompt, ~1e-3 tolerance (gates model_forward.cu before anything builds on it)
- [ ] naive_generate.cu — FP32 end-to-end driver: tokenize → model_forward → sample → loop to EOS/max_tokens. The correctness reference for every later kernel and the first thing that runs on the downloaded model. Note: at 0.6B the lm_head matmul (hidden × 151k vocab) is typically the dominant decode cost — profile it before assuming attention is.
- [ ] kv_cache_append.cu — store K/V per token once
- [ ] attention_ops.cu — GQA KV split, sliding window mask, RoPE cache + scaling (YaRN/NTK), decode-friendly layout; prerequisite for the attention kernels below
- [ ] attention_prefill.cu — process the prompt
- [ ] attention_decode.cu — one new token against cached K/V
- [ ] fused_decode_block.cu — memory-bound decode loop fused: residual+RMSNorm into QKV, RoPE into the QKV load, SiLU×Mul into the MLP up-projection
- [ ] CUDA Graphs / buffer reuse — kill launch + cudaMalloc overhead
- [ ] flash_attention.cu — Flash Attention 2/3, tiled IO-aware kernel
- [ ] paged_kv_cache.cu — vLLM-style paging, fixes fragmentation
- [ ] context_mgmt.cu — truncation + rolling window against the KV budget (the hard constraint at 4GB), prompt compression later
- [ ] prefix_cache.cu — reuse KV cache across requests sharing a prompt
- [ ] chunked_prefill.cu — split prefill into chunks, interleave with decode so long prompts don't stall other requests' latency (Sarathi's core idea)
- [ ] batched_decode.cu — static batching across sequences
- [ ] continuous_batching.cu — iteration-level scheduler (Orca-style), likely its own executable/API layer

### Quantization track (parallel to the core path; starts once naive_generate works)
- [ ] quant_ptq.cu — offline PTQ: per-channel scales/zero-point, RTN baseline at INT8 then INT4 (calibration to a real dataset, not just identity)
- [ ] quantized_matmul.cu — INT8 weight+activation matmul via DP4A (sm_75 path; no tensor cores on GTX 1650, so skip FP8/FP16-tensor-core variants)
- [ ] quant_loader.cu — read safetensors, quantize + repack weights to INT8/INT4 layout in the loader (touches model format)
- [ ] quant_validate.cu — sampling parity + perplexity vs FP32, and vs the llama.cpp Q8_0/Q4_K_M GGUF baseline of the same model

---

### Know conceptually, implement if time allows
- [ ] speculative_decode.cu — draft/verify with a small model
- [ ] kv_transfer / pd_disaggregation — split prefill and decode across separate GPU pools; understand *why* (prefill is compute-bound, decode is memory-bandwidth-bound — colocating them makes each starve the other) even without hand-rolling the NIXL/RDMA transfer layer
- [ ] tensor_parallel.cu — split weights across GPUs
- [ ] pipeline_parallel.cu — layer sharding for extreme scale
- [ ] stream_scheduler.cu — DAG scheduling, overlap compute across streams
- [ ] gpu_tokenizer.cu — avoid CPU sync per token
- [ ] kv_swap.cu — preemption/swapping when KV cache overflows
- [ ] layer_offload.cu — stream inactive weights/layers to CPU RAM (AirLLM / llama.cpp --n-gpu-layers style) to run above 4GB; decode becomes PCIe-bandwidth-bound
- [ ] kv_cache_quant.cu — quantizing the cache itself (INT8/FP8), separate from weight quantization above
- [ ] smoothquant_awq.cu — activation-aware quantization: per-channel re-scaling to suppress outliers, beats RTN (needs a calibration set)
- [ ] quant_qat.cu — quantization-aware training (fake-quant in the forward), follow-on once the training pipeline works
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
