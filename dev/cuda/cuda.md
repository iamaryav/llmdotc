#### Training
- residual_forward.cu
- gelu_forward.cu
- gelu_backward.cu
- adamw.cu
- crossentropy_forward.cu
- crossentropy_softmax_backward.cu
- encoder_forward.cu
- encoder_backward.cu
- softmax_forward.cu
- layernorm_forward.cu
- layernorm_backward.cu
- matmul_backward_bias.cu
- matmul_forward.cu
- trimat_forward.cu
- matmul_backward.cu
- fused_residual_forward.cu
- classifier_fused.cu
- global_norm.cu
- attention_forward.cu
- attention_backward.cu
- permute.cu
- nccl_all_reduce.cu


------------------------------------------------------------------------

#### Infernece
- kv_cache_append.cu
- attention_prefill.cu
- attention_decode.cu
- sampler.cu

1. KV cache — biggest conceptual and practical inference win.
2. Incremental decoding — make attention operate on one new token against cached keys/values.
3. GPU sampling — avoid copying logits to CPU every token.
4. CUDA Graphs / buffer reuse — reduce launch and allocation overhead.
5. Batching — process multiple decoding sequences together.
6. Quantization — harder, because it touches weight format, matmul kernels, accuracy, and loading.
7. Continuous batching / serving scheduler — build this last, likely as a separate executable or lightweight API layer.

------------------------------------------------------------------------
