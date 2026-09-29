import math
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

# globals must be constexpr to be used inside a @triton.jit kernel
GELU_SCALING_FACTOR = tl.constexpr(math.sqrt(2.0 / math.pi))


# python implementation 
def gelu_forward_python(inp):
    out = [0.0] * len(inp)
    for i in range(len(inp)):
        x = inp[i]
        cube = 0.044715 * x * x * x
        out[i] = 0.5 * x * (1.0 + math.tanh(GELU_SCALING_FACTOR.value * (x + cube)))
    return out


# pytorch implementation
def gelu_forward_pytorch(x):
    # x ->  B, T, C
    return F.gelu(x, approximate="tanh")


# triton implementation
# gelu forward: out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# BLOCK_SIZE = number of elements per block to process, not threads (threads come from num_warps, default 4 -> 128)
# Block_size in cuda is number of thread in each block both have diff nomenclature
@triton.jit # __global__
def gelu_forward_kernel(out_ptr, inp_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) # blockIdx.x
    # blockIdx.x * blockDim.x + threadIdx.x for the whole block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N # same as if (i < N) in CUDA
    # 128-bit loads/stores automatically if pointers and N are divisible by 16
    x = tl.load(inp_ptr + offsets, mask=mask)
    cube = 0.044715 * x * x * x
    # tl has no tanh, use the libdevice one (same tanhf as the CUDA kernel)
    out = 0.5 * x * (1.0 + libdevice.tanh(GELU_SCALING_FACTOR * (x + cube)))
    tl.store(out_ptr + offsets, out, mask=mask)


def gelu_forward(inp, block_size=1024):
    out = torch.empty_like(inp)
    N = out.numel()
    grid = (triton.cdiv(N, block_size),)
    gelu_forward_kernel[grid](out, inp, N, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    # validate against torch and benchmark
    B, T, C = 8, 1024, 768
    N = B * T * C

    inp = torch.randn(B, T, C, device="cuda", dtype=torch.float32)
    ref = gelu_forward_pytorch(inp)

    for block_size in [32, 64, 128, 256, 512, 1024]:
        out = gelu_forward(inp, block_size)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    inp_list = inp.view(-1).tolist()
    start = time.perf_counter()
    out_python = gelu_forward_python(inp_list)
    python_ms = (time.perf_counter() - start) * 1e3
    out_python = torch.tensor(out_python, dtype=torch.float32, device="cuda").view(B, T, C)
    torch.testing.assert_close(out_python, ref, atol=1e-5, rtol=1e-5)

    # print the first few values side by side
    for i in range(5):
        print(f"python {out_python.view(-1)[i].item():.6f} | torch {ref.view(-1)[i].item():.6f} | triton {out.view(-1)[i].item():.6f}")
    print("All result matched. Starting benchmarks. \n")

    # 1 read + 1 write, 4 bytes each
    memory_ops = N * 2 * 4

    # python loop is too slow for do_bench, timed once above
    print(f"python loop      | time {python_ms:.4f} ms | bandwidth {memory_ops / python_ms / 1e6:.2f} GB/s")

    torch_ms = triton.testing.do_bench(lambda: gelu_forward_pytorch(inp))
    print(f"torch            | time {torch_ms:.4f} ms | bandwidth {memory_ops / torch_ms / 1e6:.2f} GB/s")

    for block_size in [32, 64, 128, 256, 512, 1024]:
        ms = triton.testing.do_bench(lambda: gelu_forward(inp, block_size))
        bandwidth = memory_ops / ms / 1e6
        print(f"triton bs {block_size:4d}   | time {ms:.4f} ms | bandwidth {bandwidth:.2f} GB/s | speedup vs torch {torch_ms / ms:.2f}x")
