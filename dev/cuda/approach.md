# Qwen Inference Implementation Approach

Follow this table from top to bottom. `weights/qwen/config.json` and its tensor
files are the source of truth; `cuda.md` is the detailed kernel inventory.

| # | Do this next | Language / tool order | Output and completion gate |
| ---: | --- | --- | --- |
| 1 | Inspect `config.json`, tokenizer files, safetensors metadata, and all tensor names/shapes/dtypes. Confirm tied weights and Q/K norm behavior from the Transformers implementation. | Python → `safetensors` → Transformers | A tensor manifest. Current config: Qwen3, 28 layers, hidden 1024, 16 Q heads, 8 KV heads, head dim 128, MLP 3072, BF16, tied lm-head. |
| 2 | Calculate the GPU memory budget: weights, activations, workspace, and KV cache at 1K/4K/8K context. | Python | Memory report. Current BF16 KV cache is `28 × 8 × 128 × 2(K/V) × 2 bytes = 112 KiB/token`; about 448 MiB at 4K. |
| 3 | Load the exact checkpoint in PyTorch. Create fixed prompt cases and save input IDs, selected logits, greedy output IDs, and optional layer outputs. | Python → PyTorch → Transformers | Reproducible reference artifacts. This remains the oracle for all CUDA work. |
| 4 | Benchmark the PyTorch model with fixed prompt/output lengths and warm-up policy. Record GPU, CUDA version, TTFT, TPOT, and throughput. | Python → PyTorch profiler | Baseline benchmark report. |
| 5 | Write `tools/convert_hf_to_bin.py`. Define `model_config.bin` and indexed `weights.bin` with name, dtype, rank, shape, offset, byte length, alias, and format version. | Python | Deterministic converted model files. Deduplicate embedding/lm-head only after confirming they are tied. |
| 6 | Write a host-side runtime loader that validates file bounds, tensor names, dtypes, and shapes before GPU upload. | Python format → C++ loader | Loader prints/validates every expected tensor. Reimplement the converter in C++ only after Python format is stable. |
| 7 | Implement residual add and test it against PyTorch on random and awkward tensor sizes. | CUDA C++ → Python/PyTorch test | `residual_forward.cu` passes numerical test and compute-sanitizer. |
| 8 | Implement **RMSNorm** (not LayerNorm for Qwen), then embedding lookup. | CUDA C++ → Python/PyTorch test | Both match PyTorch within declared tolerance. |
| 9 | Build a reusable linear-layer wrapper using cuBLAS/cuBLASLt. Upload and read BF16 weights; begin with FP32 activation/accumulation. | C++ runtime → CUDA C++ → cuBLAS/cuBLASLt → PyTorch test | Linear layer matches PyTorch. Do not write custom GEMM yet. |
| 10 | Implement SiLU, SwiGLU, softmax, RoPE, and Q/K normalization as separate tested kernels. | CUDA C++ → Python/PyTorch tests | Each primitive has parity tests, model-shaped cases, and no memory errors. |
| 11 | Implement GQA attention without a KV cache. | CUDA C++ → Python/PyTorch test | Attention output matches PyTorch on short sequences. |
| 12 | Assemble one Qwen decoder block: RMSNorm → QKV/QK norm/RoPE/GQA → projection/residual; RMSNorm → SwiGLU MLP → residual. | C++ runtime → CUDA C++ → PyTorch test | One-layer activations match saved PyTorch activations. |
| 13 | Loop all 28 layers, add final RMSNorm and tied lm-head/vocabulary projection. | C++ runtime → CUDA C++ → PyTorch test | Full-forward logits match PyTorch for fixed token IDs. |
| 14 | Implement tokenizer parity with `AutoTokenizer`, then greedy (`argmax`) generation with EOS and max-token handling. | Python/AutoTokenizer → C++ tokenizer → Python parity harness | Token IDs are exact; several greedy generations match the reference. |
| 15 | Add temperature, top-k, top-p, repetition penalty, and stop handling. Avoid per-token logits device-to-host copies when optimizing. | CUDA C++ sampler → C++ runtime → Python tests | Sampling tests pass; deterministic settings remain reproducible. |
| 16 | Define the KV-cache layout and implement append, prompt prefill, and one-token decode. | CUDA C++ → Python/PyTorch test | Cached and non-cached logits match; capacity is checked. |
| 17 | Benchmark prefill and decode separately. Record TTFT, TPOT, throughput, cache size, and context length. | C++ harness → Nsight Systems | Reproducible inference baseline. |
| 18 | Profile actual bottlenecks. Improve memory access, buffer reuse, and launch configuration; use CUDA Graphs only if launch overhead shows up. | Nsight Systems/Compute → CUDA C++ → benchmark | Before/after profiler evidence and benchmark numbers. |
| 19 | Fuse only measured low-arithmetic-intensity paths: residual/RMSNorm, RoPE/QK handling, or SwiGLU. Compare with the unfused path. | CUDA C++ → Python parity → benchmark | Same output tolerance and measured improvement. |
| 20 | Study tiled/Flash-style attention and compare Triton or CUTLASS implementations for selected kernels. | CUDA C++ → Triton/CUTLASS → benchmark | Comparison report; retain the simplest implementation that meets the target. |
| 21 | Add offline INT8 quantization, validate logits/generation quality, then evaluate INT4/repacking. | Python converter → CUDA C++ runtime → PyTorch quality test | Quantized format, quality report, and speed/memory comparison. |
| 22 | Compare the final runner with PyTorch and llama.cpp on identical settings. | Python/C++ harness → benchmark table | Fair benchmark table with commands and hardware metadata. |
| 23 | Add serving features only after single-request decode works: paged KV, context management, prefix cache, chunked prefill, static batching, then continuous batching. | C++ scheduler → CUDA C++ kernels → benchmark | Multi-request correctness and latency/throughput tests. |

## Rules for every CUDA step

| Always do | Why |
| --- | --- |
| Keep the PyTorch reference | It catches semantic and numerical errors quickly. |
| Write a small test before optimizing | A fast wrong kernel is hard to debug. |
| Run compute-sanitizer regularly | Find invalid accesses and races early. |
| Profile before fusion or custom GEMM | Optimize measured bottlenecks, not guesses. |
| Record inputs, precision, GPU, and metrics | Benchmarks stay reproducible and comparable. |
