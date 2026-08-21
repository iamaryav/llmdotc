/*
Kernels for crossentropy forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt crossentropy_forward.cu -o crossentropy_forward

version 1 is naive port from CPU code to GPU, one thread per token
./crossentropy_forward 1
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "common.h"


void crossentropy_forward_cpu(float* losses, const float* probs, const int* targets, int B, int T, int V) {
    // calculates -log(probs)

    // B * T * V for each batch (skips over B size)
    // T * V for each token inside a batch (skips over t tokens)
    // B * T * v + t * v points to the start of the probablity vector of the t_th token
    // for target
    // B * T + t
    // simply, math of pointing to the start of right probablity vector and right target index

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

// -------------------------------------------------------------------------
// GPU kernels

__global__ void crossentropy_forward_kernel1(float* losses, const float* probs, const int* targets, int B, int T, int V){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T) {
        int b = i / T;
        int t = i % T;
        const float* probs_bt = probs + b * T * V + t * V;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
    }
}


// -------------------------------------------------------------------------
// kernel launcher

void crossentropy_forward1(float* losses,
                            const float* probs, const int* targets,
                            int B, int T, int V,
                            const int block_size) {
    int N = B * T;
    int grid_size = ceil_div(N, block_size);
    crossentropy_forward_kernel1<<<grid_size, block_size>>>(losses, probs, targets, B, T, V);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void crossentropy_forward(int kernel_num,
                          float* losses,
                          const float* probs, const int* targets,
                          int B, int T, int V,
                          const int block_size) {
    switch (kernel_num) {
        case 1:
            crossentropy_forward1(losses, probs, targets, B, T, V, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

// -------------------------------------------------------------------------

int main (int argc, char** argv) {
    int B = 8;
    int T = 1024;
    int V = 50257;

    cudaCheck(cudaSetDevice(0));

    float* out = (float*)malloc(B * T * sizeof(float));
    float* probs =  make_random_float01(B * T * V);
    int* targets = make_random_int(B * T, V);

    // move to GPU
    float* d_out;
    float* d_probs;
    int* d_targets;
    cudaCheck(cudaMalloc(&d_out, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_probs, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_targets, B * T * sizeof(int)));

    cudaCheck(cudaMemcpy(d_probs, probs, B * T * V * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // read kernel for command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);


    //check the calculation
    crossentropy_forward_cpu(out, probs, targets, B, T, V);

    // kernel at different sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Using the block size %d\n", block_size);
        crossentropy_forward(kernel_num, d_out, d_probs, d_targets, B, T, V, block_size);
        validate_result(d_out, out, "out", B * T, 1e-5f);
    }

    printf("All result match, starting benchmarks. \n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, crossentropy_forward, kernel_num, d_out, d_probs, d_targets, B, T, V, block_size);

        printf("block_size %4d | time %.4f ms | per token %.2f ns\n", block_size, elapsed_time, elapsed_time * 1e6f / (B * T));
    }


    // free memory
    free(out);
    free(probs);
    free(targets);
    cudaFree(d_out);
    cudaFree(d_probs);
    cudaFree(d_targets);
    return 0;
}
