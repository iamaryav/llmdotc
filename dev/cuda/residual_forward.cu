#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#include "common.h"



// ------------------------------------------------------------
// cpu code
// x = x + causal_attn(ln(x))
// two input array and need to add them

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, const int N){
	for(int i = 0; i < N; i++){
		out[i] = inp1[i] + inp2[i];
	}
}

// ------------------------------------------------------------
// gpu kernels 

__global__ void residual_forward_kernel1(float* out, const float* inp1, const float* inp2, const int N){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N){
        out[idx] = inp1[idx] + inp2[idx];
	}
}

__global__ void residual_forward_kernel2(){


}

// ------------------------------------------------------------
// run code

int main(int argc, char** argv){

	// int B = 8;
	// int B = 32;
	// int T = 1024;
	// int C = 768;
	int B = 64;
	int T = 2048;
	int C = 1024;
	int N = B * T * C;

	// creating memory on host and initializing with 
	// random 
	float* out = (float*)malloc(N * sizeof(float));
	float* c_out = (float*)malloc(N * sizeof(float));
	float* inp1 = make_random_float(N);
	float* inp2 = make_random_float(N);

	// running the kernel on cpu
	residual_forward_cpu(c_out, inp1, inp2, N);

	for (int i = 0; i < 10; i++){
		printf("%f + %f = %f\n", inp1[i], inp2[i], c_out[i]);

	}
    // launch the kernel
    // move data to gpu
    // use floatX for bfloat16 in common.h file
    
    float* d_out;
    float* d_inp1;
    float* d_inp2;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_inp1, N * sizeof(float));
    cudaMalloc(&d_inp2, N * sizeof(float));

    cudaMemcpy(d_inp1, inp1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp2, inp2, N * sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 128;
    // int grid_size = 4;
    int grid_size = (N + block_size - 1) / block_size;
    residual_forward_kernel1<<<grid_size, block_size>>>(d_out, d_inp1, d_inp2, N);
    cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU kernel is calculating....\n");
	for (int i = 0; i < 10; i++){
		printf("%f + %f = %f\n", inp1[i], inp2[i], out[i]);
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
