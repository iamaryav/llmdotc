#include<stdio.h>
#include<stdlib.h>


#include "common.h"





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

__global__ void gelu_backward_kernel1(floatX* dinp, const floatX* inp, const float* dout, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x
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

    }
}
//-------------------------------------------------------------

int main(int argc, char** argv) {
    // setup main

    // int B = 8;
    // int T = 1024;
    // int C = 768;
    
    int B = 2;
    int T = 8;
    int C = 4;
    int N = B * T * C;

    float* dinp = (float*)malloc(N * sizeof(float));
    float* dout = make_random_float(N);
    float* inp = make_random_float(N);

    gelu_backward_cpu(dinp, inp, dout, N);

    for (int i = 0; i < N; i++) { 
        printf("inp: %f, | dout: %f | dinp: %f\n", inp[i], dout[i], dinp[i]);
    }

    return 0;
}
