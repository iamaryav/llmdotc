/*
Kernels for AdamW optimizer.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt adamw.cu -o adamw

version 1 is naive GPU kernel
./adamw 1

version 2 is optimized GPU kernel using registers and fused multiply-add
./adamw 2
*/

#include<stdio.h>
#include<stdlib.h>

#include "common.h"

void adamw_cpu(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_params, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8, float weight_decay=0.0) {

    // calculate the m_t, v_t, bias correction, weight decay
    // for each params we have to calculate the adamw
    // by how much we should nudge the weight and which direction
    for (int i = 0; i < num_params; i++){
        float param = params_memory[i];
        float grad = grads_memory[i];
        // first moment (momentum)
        float m = beta1 * m_memory[i] + (1.0f - beta1) * grad;
        // second moment (RMSprop)
        float v = beta2 * v_memory[i] + (1.0f - beta2) * grad * grad;

        // bias correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        //updates
        m_memory[i] = m;
        v_memory[i] = v;
        params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);

    }

}

//---------------------------------------------------------------------------------------------------
//GPU kernel

__device__ inline float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

// naive adamw GPU kernel
// this is naive because we are not fusing operations just using simple calculation flow
// with more threads can be speedup with combining multiple opernation together to avoid
// multiple memory movements
__global__ void adamw_kernel1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_params, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {

    // first we need thread idx
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_params) return;

    // first moment calcualtion
    float m = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
    // second moment calculation (RMSprop)
    float v = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];

    m_memory[i] = m;
    v_memory[i] = v;

    // bias corrected moments
    float m_hat = m / beta1_correction;
    float v_hat = v / beta2_correction;

    // it's time to update
    params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params_memory[i]);
}

// slighlty optimized adamw kernel by using below tricks
// * loading data that is accessed more than once into registers,
// * using optimized linear interpolation for the moment updates.
// use of fused multiply addition
//
__global__ void adamw_kernel2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_params, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {

    // To find which thread am I?
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_params) return;
    // grads memory, m and v memory we can load it before calculation
    float grad = grads_memory[i];
    float m = m_memory[i];
    float v = v_memory[i];

    // update the first moment
    // using fused multiplication and addition and saving time 
    // because gpus has dedicated hard for this instead of calculating multiplication and addition
    // seperately
    m = lerp(grad, m, beta1);
    m_memory[i] = m;

    // updated the second moment
    v = lerp(grad * grad, v, beta2);
    v_memory[i] = v;

    m /=  beta1_correction; // m_hat
    v /=  beta2_correction; // v_hat

    params_memory[i] -= learning_rate * ( m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

//---------------------------------------------------------------------------------------------------
//
//
//kernel launcher
void adamw_dispatch1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    // block size, thread per block
    unsigned int block_size = 512; // number of thread per block
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    adamw_kernel1<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory,
                                              num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}


void adamw_dispatch2(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, long num_parameters, float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay) {
    // block size, thread per block
    unsigned int block_size = 512; // number of thread per block
    unsigned int num_blocks = ceil_div(num_parameters, (long) block_size);
    adamw_kernel2<<<num_blocks, block_size>>>(params_memory, grads_memory, m_memory, v_memory,
                                              num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}


// gpu kernel dispatch
void adamw(int kernel_num, float* params_memory, const float* grads_memory, float* m_memory, float* v_memory,
           int t, long num_parameters, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float eps=1e-8,
           float weight_decay=0.0) {
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    switch (kernel_num) {
        case 1:
            adamw_dispatch1(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        case 2:
            adamw_dispatch2(params_memory, grads_memory, m_memory, v_memory, num_parameters,
                            learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
            break;
        default:
            printf("Invalid kernel number \n");
            exit(1);
    }
}

//---------------------------------------------------------------------------------------------------

int main(int argc, char** argv){

    float* params_memory;
    float* grads_memory;
    float* m_memory;
    float* v_memory;

    // intializing the hyper params
    const int t = 10;
    const long num_params = 1048576;
    const float learning_rate = 1e-3;
    const float beta1 = 0.9;
    const float beta2 = 0.999;
    float eps = 1e-8;
    float weight_decay = 0.0;

    // initializing these with random values
    params_memory = make_random_float(num_params);
    grads_memory = make_random_float(num_params);
    m_memory = make_random_float(num_params);
    v_memory = make_random_float01(num_params);

    // move to GPU
    float* d_params_memory;
    float* d_grads_memory;
    float* d_m_memory;
    float* d_v_memory;

    cudaCheck(cudaMalloc(&d_params_memory, num_params * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grads_memory, num_params * sizeof(float)));
    cudaCheck(cudaMalloc(&d_m_memory, num_params * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v_memory, num_params * sizeof(float)));

    cudaCheck(cudaMemcpy(d_params_memory, params_memory, num_params * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grads_memory, grads_memory, num_params * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m_memory, m_memory, num_params * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v_memory, v_memory, num_params * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // calculate the time taken by code ran on CPU
    clock_t start = clock();
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_params);
    clock_t end = clock();

    double elpased_time_cpu = (double)(end - start) / CLOCKS_PER_SEC;

    // calculate the gpu version using default params
    adamw(kernel_num, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_params);

    //compare
    printf("checking correctness...\n");
    printf("parameters:\n");
    validate_result(d_params_memory, params_memory, "params_memory", num_params);
    printf("first momoent:\n");
    validate_result(d_m_memory, m_memory, "m_memory", num_params);
    printf("second momoent:\n");
    validate_result(d_v_memory, v_memory, "v_memory", num_params);

    printf("All results matched \n");

    // now benchemarking the kernel
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, adamw, kernel_num, d_params_memory, d_grads_memory, d_m_memory, d_v_memory, t, num_params, learning_rate, beta1, beta2, eps, weight_decay);
    printf("time gpu %.4f ms\n", elapsed_time);
    printf("time cpu %.4f ms\n", elpased_time_cpu * 1000.0);


    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);
    cudaCheck(cudaFree(d_params_memory));
    cudaCheck(cudaFree(d_grads_memory));
    cudaCheck(cudaFree(d_m_memory));
    cudaCheck(cudaFree(d_v_memory));

    return 0;
}
