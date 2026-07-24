/*
Kernels for GELU backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt gelu_backward.cu -o gelu_backward

If encountering "error: identifier \"M_PI\" is undefined", add the following lines to the top of the file:

#define _USE_MATH_DEFINES
#include <math.h>  OR  #include <cmath>

version 1 is naive CPU port
./gelu_backward 1

version 2 is bfloat16 with the Packed128 data structure
./gelu_backward 2
*/



#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>


#include "common.h"
//#define ENABLE_BF16





//-------------------------------------------------------------
// CPU code ref

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N){
    // approximation formula 
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_args = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_args);
        float coshf_out = coshf(tanh_args);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }

}

//-------------------------------------------------------------
// GPU Kernels

__global__ void gelu_backward_kernel1(floatX* dinp, const floatX* inp, const floatX* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_args = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_args);
        float coshf_out = coshf(tanh_args);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (floatX)(local_grad * (float)dout[i]);
    }
}

__global__ void gelu_backward_kernel2(floatX* dinp, const floatX* inp, const float* dout, const int N) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N) {
        x128 packed_dinp;
        // read 16 bytes from the given address
        x128 packed_inp = load128cs(inp + i);
        x128 packed_dout = load128cs(dout + i);

        for (int k = 0; k < packed_inp.size; ++k) {
            float x = (float)packed_inp[k];
            float cube = 0.044715f * x * x * x;
            float tanh_args = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = tanhf(tanh_args);
            float coshf_out = coshf(tanh_args);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
        }
        // write 16 byte from packed_dinp to the location starts from dinp + i
        store128(dinp + i, packed_dinp);
    }
}
//-------------------------------------------------------------
// kernel launcher

void gelu_backward1(floatX* dinp, const floatX* inp, const floatX* dout, const int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    gelu_backward_kernel1<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward2(floatX* dinp, const floatX* inp, const floatX* dout, const int N, const int block_size){
    const int grid_size = ceil_div(N, block_size * x128::size);
    gelu_backward_kernel2<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch


void gelu_backward(const int kernel_num, floatX* dinp, const floatX* inp, const floatX* dout, const int N, const int block_size){

    switch(kernel_num) {
        case 1: 
            gelu_backward1(dinp, inp, dout, N, block_size);
            break;
        case 2: 
            gelu_backward2(dinp, inp, dout, N, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

//-------------------------------------------------------------

int main(int argc, char** argv) {
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int N = B * T * C;

    // create host memory of random numbers
    float* dinp = (float*)malloc(N * sizeof(float));
    float* dout = make_random_float(N);
    float* inp = make_random_float(N);

    // read kernel num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }

    printf("Using kernel %d\n", kernel_num);

    // correctness of kernel in CPU
    gelu_backward_cpu(dinp, inp, dout, N);

    // GPU
    floatX* d_dinp;
    floatX* d_dout;
    floatX* d_inp;
    cudaCheck(cudaMalloc(&d_dinp, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, N * sizeof(floatX)));

    cudaCheck(memcpy_convert(d_dout, dout, N));
    cudaCheck(memcpy_convert(d_inp, inp, N));

    // kernel at different block_size
    // block_size: no of thread in a block
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size: %d\n", block_size);
        gelu_backward(kernel_num, d_dinp, d_inp, d_dout, N, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", N, tol);
    }

    printf("All results math. Starting benchmarks.\n\n");

    for(int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        
        // elapsed time is in ms
        float elapsed_time = benchmark_kernel(repeat_times, gelu_backward, kernel_num, d_dinp, d_inp, d_dout, N, block_size);

        // bandwidth achieved
        // for each (B, T, C) output element, we do 2 reads and 1 write, 4 bytes each
        long memory_ops = N * 3 * sizeof(floatX);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }
    
    // free memory
    free(dinp);
    free(dout);
    free(inp);
    cudaFree(d_dinp);
    cudaFree(d_dout);
    cudaFree(d_inp);
    return 0;
}
