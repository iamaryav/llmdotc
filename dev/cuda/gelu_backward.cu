#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>



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


void gelu_backward(){

}

//-------------------------------------------------------------

int main(int argc, char** argv) {
    // setup main

    // int B = 8;
    // int T = 1024;
    // int C = 768;
    
    int B = 2;
    int T = 4;
    int C = 4;
    int N = B * T * C;

    float* dinp = (float*)malloc(N * sizeof(float));
    float* dout = make_random_float(N);
    float* inp = make_random_float(N);

    gelu_backward_cpu(dinp, inp, dout, N);

    for (int i = 0; i < N; i++) { 
        printf("inp: %8.4f, | dout: %8.4f | dinp: %8.4f\n", inp[i], dout[i], dinp[i]);
    }

    floatX* d_dinp;
    floatX* d_dout;
    floatX* d_inp;

    cudaCheck(cudaMalloc(&d_dinp, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, N * sizeof(floatX)));

    // cudaMemcpy(d_dinp, dinp, N * sizeof(floatX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, dout, N * sizeof(floatX), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp, inp, N * sizeof(floatX), cudaMemcpyHostToDevice);

    // grid size 2, block size 32
    gelu_backward_kernel1<<<2, 32>>>(d_dinp, d_inp, d_dout, N);
    float* dh_inp = (float*)malloc(N * sizeof(float));


    cudaMemcpy(dh_inp, d_dinp, N * sizeof(floatX), cudaMemcpyDeviceToHost);


    printf("GPU output\n");
    for (int i = 0; i < N; i++) { 
        printf("inp: %8.4f, | dout: %8.4f | dhinp: %8.4f\n", inp[i], dout[i], dh_inp[i]);
    }


    // launching kernel first normal way 
    // then through kernle launcher
    
    free(dinp);
    free(dout);
    free(inp);
    cudaFree(d_dinp);
    cudaFree(d_dout);
    cudaFree(d_inp);
    cudaFree(dh_inp);

    return 0;
}
