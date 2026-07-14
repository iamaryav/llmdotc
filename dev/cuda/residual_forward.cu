/*
Requirements: CUDA Toolkit, NVIDIA GPU
Check compute capability: nvidia-smi --query-gpu=compute_cap --format=csv,noheader
FP16: sm_53+   |   BF16: sm_80+

Build & Run:
nvcc -O3 --use_fast_math residual_forward.cu -o residual_forward -lcublas -lcublasLt && ./residual_forward 1

Mixed precision (pass -D flag or uncomment below):
nvcc -O3 --use_fast_math -DENABLE_FP16 -arch=sm_70 residual_forward.cu -o residual_forward -lcublas -lcublasLt  # FP16
nvcc -O3 --use_fast_math -DENABLE_BF16 -arch=sm_80 residual_forward.cu -o residual_forward -lcublas -lcublasLt  # BF16
# precedence: BF16 > FP16 > fp32
*/

#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>

// #define ENABLE_BF16  // uncomment for BF16 (requires sm_80+)
// #define ENABLE_FP16  // uncomment for FP16
#include "common.h"



// ------------------------------------------------------------
// cpu code

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, const int N){
	for(int i = 0; i < N; i++){
		out[i] = inp1[i] + inp2[i];
	}
}

// ------------------------------------------------------------
// gpu kernels 

__global__ void residual_forward_kernel1(floatX* out, const floatX* inp1, const floatX* inp2, const int N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N){
        out[idx] = inp1[idx] + inp2[idx];
	}
}

__global__ void residual_forward_kernel2(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx < N) {
        x128 packed_out;
        x128 packed_inp1 = load128cs(inp1 + idx);
        x128 packed_inp2 = load128cs(inp2 + idx);
        for (int k = 0; k < packed_inp1.size; ++k) {
            packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
        }
        store128(out + idx, packed_out);
    }
}

// ------------------------------------------------------------
// kernel launcher

void residual_forward1(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    residual_forward_kernel1<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

void residual_forward2(floatX* out, const floatX* inp1, const floatX* inp2, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size * x128::size);
    residual_forward_kernel2<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}


// kernel version dispatch
void residual_forward(int kernel_num,
                      floatX* out,
                      const floatX* inp1,
                      const floatX* inp2,
                      int N,
                      int block_size) {
    switch (kernel_num) {
        case 1:
            residual_forward1(out, inp1, inp2, N, block_size);
            break;
        case 2:
            residual_forward2(out, inp1, inp2, N, block_size);
            break;
        default:
            printf("Invalid kernel number \n");
            exit(1);
    }
}

// ------------------------------------------------------------
// run code

int main(int argc, char** argv){

	int B = 8;
	int T = 1024;
	int C = 768;
	int N = B * T * C;

	// creating memory on host and initializing with random
	float* out = (float*)malloc(N * sizeof(float));
	float* c_out = (float*)malloc(N * sizeof(float));
	float* inp1 = make_random_float(N);
	float* inp2 = make_random_float(N);

    
    floatX* d_out;
    floatX* d_inp1;
    floatX* d_inp2;
    cudaCheck(cudaMalloc(&d_out, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp1, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp2, N * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp1, inp1, N));
    cudaCheck(memcpy_convert(d_inp2, inp2, N));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }

    printf("Using kernel %d\n", kernel_num);

	// running the kernel on cpu
	residual_forward_cpu(c_out, inp1, inp2, N);

    // time the kernel at the different block size
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        residual_forward(kernel_num, d_out, d_inp1, d_inp2, N, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
    float tol = 1e-5;
#else
    float tol = 1e-2f;
#endif
        validate_result(d_out, c_out, "out", N, tol);
    }

    printf("All result matched. Starting benchmarks. \n\n");


    // benchmarking
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j]; 

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, residual_forward, kernel_num, d_out, d_inp1, d_inp2, N, block_size);

        // napkin math time
        // for each (B, T, C) output, we do 2 read and 1 write, 4 bytes each
        long memory_ops = B * T * C * 3 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s \n", block_size, elapsed_time, memory_bandwidth);
    }

    cudaFree(d_inp1);
    cudaFree(d_inp2);
    cudaFree(d_out);
    free(inp1);
    free(inp2);
    free(out);
    free(c_out);

	return 0;
}
