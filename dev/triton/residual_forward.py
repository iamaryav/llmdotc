import torch
import triton
import triton.language as tl


# residual forward: out = inp1 + inp2
@triton.jit # __global__
def residual_forward_kernel(out_ptr, inp1_ptr, inp2_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) # blockIdx.x
    # blockIdx.x * blockDim.x + threadIdx.x
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N # same as if (idx < N) in CUDA
    # read the input values
    inp1 = tl.load(inp1_ptr + offsets, mask)
    inp2 = tl.load(inp2_ptr + offsets, mask)
    tl.store(out_ptr + offsets, inp1 + inp2, mask=mask)


def residual_forward(inp1, inp2, block_size=1024):
    out = torch.empty_like(inp1)
    N = out.numel()
    grid = (triton.cdiv(N, block_size),)
    residual_forward_kernel[grid](out, inp1, inp2, N, BLOCK_SIZE=block_size)
    return out


# validate against torch and benchmark
B, T, C = 8, 1024, 768
N = B * T * C

inp1 = torch.randn(N, device="cuda", dtype=torch.float32)
inp2 = torch.randn(N, device="cuda", dtype=torch.float32)
ref = inp1 + inp2

for block_size in [32, 64, 128, 256, 512, 1024]:
    out = residual_forward(inp1, inp2, block_size)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

# print the first few values side by side
for i in range(5):
    print(f"torch {ref[i].item():.6f} | triton {out[i].item():.6f}")
print("All result matched. Starting benchmarks. \n")

# 2 reads + 1 write, 4 bytes each
memory_ops = N * 3 * 4

torch_ms = triton.testing.do_bench(lambda: inp1 + inp2)
print(f"torch            | time {torch_ms:.4f} ms | bandwidth {memory_ops / torch_ms / 1e6:.2f} GB/s")

for block_size in [32, 64, 128, 256, 512, 1024]:
    ms = triton.testing.do_bench(lambda: residual_forward(inp1, inp2, block_size))
    bandwidth = memory_ops / ms / 1e6
    print(f"triton bs {block_size:4d}   | time {ms:.4f} ms | bandwidth {bandwidth:.2f} GB/s | speedup vs torch {torch_ms / ms:.2f}x")

