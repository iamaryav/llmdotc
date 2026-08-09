// adamw optimizer

#include<stdio.h>
#include<stdlib.h>

#inlclude "common.h"

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
__global__ void adamw_kernel1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_params, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float beta1_correction, float beta2_correction, float eps=1e-8, float wegith_decay=0.0) {

    // first we need thread idx
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_params) return;

    // first moment calcualtion
    m_memory[i] = beta1 * m_memory[i] + (1.0f - beta1) * grads_memory[i];
    // second moment calculation (RMSprop)
    v_memory[i] = beta2 * v_memory[i] + (1.0f - beta2) * grads_memory[i] * grads_memory[i];

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
__global__ void adamw_kernel1(float* params_memory, const float* grads_memory, float* m_memory, float* v_memory, int t, long num_params, float learning_rate=1e-3, float beta1=0.9, float beta2=0.999, float beta1_correction, float beta2_correction, float eps=1e-8, float wegith_decay=0.0) {

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
    v = lerp(grad, v, beta2);
    v_memory = v;

    m /=  beta1_correction; // m_hat
    v /=  beta2_correction; // v_hat

    params_memory[i] -= learning_rate * ( m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}




//---------------------------------------------------------------------------------------------------


int main(int argc, char** argv){

    float* params_memory;
    float* grads_memory;
    float* m_memory;
    float* v_memory;

    // intializing the hyper params
    const int t = 5;
    const long num_params = 1024;
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

    // before calculation 
    for(int i = 0; i < 5; i++){
        printf("params_memory: %8.4f grads_memory: %8.4f, m_memory: %8.4f, v_memory: %8.4f\n", params_memory[i], grads_memory[i], m_memory[i], v_memory[i]);
    }
    adamw_cpu(params_memory, grads_memory, m_memory, v_memory, t, num_params, learning_rate, beta1, beta2, eps, weight_decay);

    // after calculation
    printf("After adamw calculation\n");
    for(int i = 0; i < 5; i++){
        printf("params_memory: %8.4f grads_memory: %8.4f, m_memory: %8.4f, v_memory: %8.4f\n", params_memory[i], grads_memory[i], m_memory[i], v_memory[i]);
    }
    
    free(params_memory);
    free(grads_memory);
    free(m_memory);
    free(v_memory);

    return 0;
}
