  1. KV cache — biggest conceptual and practical inference win.
  2. Incremental decoding — make attention operate on one new token against cached keys/values.
  3. GPU sampling — avoid copying logits to CPU every token.
  4. CUDA Graphs / buffer reuse — reduce launch and allocation overhead.
  5. Batching — process multiple decoding sequences together.
  6. Quantization — harder, because it touches weight format, matmul kernels, accuracy, and loading.
  7. Continuous batching / serving scheduler — build this last, likely as a separate executable or lightweight API
     layer.
