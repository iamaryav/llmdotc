/*
Kernels for the encoder (token + positional embedding) forward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt encoder_forward.cu -o encoder_forward

version 1 is naive port from CPU code to GPU, one thread per (b, t) token
./encoder_forward 1

version 2 parallelizes over (b, t, c), one thread per output element
./encoder_forward 2

version 3 is like version 2 but uses the Packed128 data structure
./encoder_forward 3
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// #define ENABLE_BF16  // uncomment for BF16 (requires sm_80+)
// #define ENABLE_FP16  // uncomment for FP16
#include "common.h"

//-------------------------------------------------------------------------
// CPU Code
// take the input and apply token and position embedding
// CPU code reference
// Positional encoder forward pass
void encoder_forward_cpu(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C){
    // out - B * T * C
    // input - B * T
    // embedding dim - v * C
    // B * T * C
    // B * T * c + T * c

    //c1, c2, c3
    // b1 -> t1, t2.... | b2 -> t1, t2...
    for (int b = 0; b < B; b++){
        for (int t = 0; t < T; t++){
            float* out_bt = out + b * T * C + t * C; // base address of each t
            int ix = inp[b * T + t];
            const float* wte_ix = wte + ix * C;
            const float* wpe_ix = wpe + t * C;
            for (int i = 0; i < C; i++){
                out_bt[i] = wte_ix[i] + wpe_ix[i];
            }
        }
    }
}

//-------------------------------------------------------------------------
// GPU code

__global__ void encoder_forward_kernel1(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    if (idx < N) {
        int b = idx / T;
        int t = idx % T;
        floatX* out_bt = out + b * T * C + t * C;
        int ix = inp[b * T + t];
        const floatX* wte_ix = wte + ix * C;
        const floatX* wpe_t = wpe + t * C;
        for (int i = 0; i < C; i++){
            out_bt[i] = (floatX)((float)wte_ix[i] + (float)wpe_t[i]);
        }
    }

}

__global__ void encoder_forward_kernel2(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C){
    // parallelize over B, T, C rathaer than B, T
    // use float4 and int4 packed data to minimize the memory movement
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_t = wpe + t * C + c;
        *out_btc = (floatX)((float)*wte_ix + (float)*wpe_t);
    }
}

__global__ void encoder_forward_kernel3(floatX* out, const int* inp, const floatX* wte, const floatX* wpe, int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;

        int ix = inp[b * T + t];

        floatX* out_btc = out + b * T * C + t * C + c;
        const floatX* wte_ix = wte + ix * C + c;
        const floatX* wpe_tc = wpe + t * C + c;

        x128 packed_out;
        x128 packed_wte = load128cs(wte_ix);
        x128 packed_wpe = load128cs(wpe_tc);
        #pragma unroll
        for (int k = 0; k < x128::size; k++) {
            packed_out[k] = (floatX)((float)packed_wte[k] + (float)packed_wpe[k]);
        }
        store128(out_btc, packed_out);
    }
}

// ---------------------------------------------------------------------
// kernel launcher

void encoder_forward1(floatX* out,
                      const int* inp, const floatX* wte, const floatX* wpe,
                      int B, int T, int C,
                      const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel1<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_forward2(floatX* out,
                      const int* inp, const floatX* wte, const floatX* wpe,
                      int B, int T, int C,
                      const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, block_size);
    encoder_forward_kernel2<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_forward3(floatX* out,
                      const int* inp, const floatX* wte, const floatX* wpe,
                      int B, int T, int C,
                      const int block_size) {
    const int N = B * T * C;
    const int grid_size = ceil_div(N, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size>>>(out, inp, wte, wpe, B, T, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void encoder_forward(int kernel_num,
                     floatX* out,
                     const int* inp, const floatX* wte, const floatX* wpe,
                     int B, int T, int C,
                     const int block_size) {
    switch (kernel_num) {
        case 1:
            encoder_forward1(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 2:
            encoder_forward2(out, inp, wte, wpe, B, T, C, block_size);
            break;
        case 3:
            encoder_forward3(out, inp, wte, wpe, B, T, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}


// ---------------------------------------------------------------------

void benchmark_kernels(){

}

int main(int argc, char** argv){
    int B = 8;
    int T = 1024;
    int C = 768;
    int V = 50257;
    int N = B * T * C;
    float* out;
    int* inp;
    float* wte;
    float* wpe;

    // memory allocation
    out = (float*)malloc(N * sizeof(float));
    inp = make_random_int(B * T, V);
    wte = make_random_float(V * C);
    wpe = make_random_float(T * C);

    // device
    floatX* d_out;
    int* d_inp;
    floatX* d_wte;
    floatX* d_wpe;

    // memory allocation
    cudaCheck(cudaMalloc(&d_out, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * sizeof(int)));
    cudaCheck(cudaMalloc(&d_wte, V * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_wpe, T * C * sizeof(floatX)));

    cudaCheck(cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(memcpy_convert(d_wte, wte, V * C));
    cudaCheck(memcpy_convert(d_wpe, wpe, T * C));

    // check correctness of the kernel using CPU
    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);
    int kernel_num = 2;
    if (argc > 1) {
         kernel_num = atoi(argv[1]);
    }

    printf("Using kernel num: %d\n", kernel_num);

    // number of thread in a block
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    
    for (int i = 0; i < sizeof(block_sizes) / sizeof(int); i++){
        int block_size = block_sizes[i];
        printf("Checking block size %d.\n", block_size);
        encoder_forward(kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size);

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif

        validate_result(d_out, out, "out", B * T * C, tol);

    }

    printf("All results match. Starting benchmarks.\n\n");


    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, encoder_forward,
                                              kernel_num, d_out, d_inp, d_wte, d_wpe, B, T, C, block_size
                                              );

        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }


    // free the memory
    free(out);
    free(inp);
    free(wte);
    free(wpe);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_wte));
    cudaCheck(cudaFree(d_wpe));
    return 0;
}
