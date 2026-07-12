/*
nvcc gelu_forward.cu -o gelu_forward






*/

#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>


#define ENABLE_BF16
#include "common.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// ----------------------------------------------------------------
// CPU code
// there are N values -> B * T * C
// these values are contiguous in memory
void gelu_forward_cpu(float* inp, float* out, int N){
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf( GELU_SCALING_FACTOR * (x + cube)));

    }
}

// ----------------------------------------------------------------
// GPU Kernels

__global__ void gelu_forward_kernel1(floatX* inp, floatX* out, int N){
    // each thread will do the one ops
    idx = blockIdx.x * blockdim.x + threadIdx.x;
    if (idx < N) {
        float x = out[idx];
        float cube = 0.044715f * x * x * x;
        out[idx] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}









// ----------------------------------------------------------------

int main(int argc, char** argv){

    // define setup main method  to setup gpu
    
    // int B = 8;
    // int T = 1024;
    // int C = 768;

    int B = 2;
    int T = 4;
    int C = 4;
    int N = B * T * C;

    float* inp = make_random_float(N);
    float* out;

    // inp = (float*) malloc(N * sizeof(float));
    out = (float*) malloc(N * sizeof(float));

    gelu_forward_cpu(inp, out, N);

    printf("cpu output: \n");
    for(int i = 0; i < N; i++) {
        printf("x: %f -> gelu(x): %f\n", inp[i], out[i]);
    }

    free(inp);
    free(out);
    return 0;
}
