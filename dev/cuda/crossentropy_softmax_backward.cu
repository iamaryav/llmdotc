#include<stdio.h>
#include<stdlib.h>

#include "common.h"


// ------------------------------------------------------------------------------
// cross entropy backward
//
//

// cpu code ref

void crossentropy_softmax_backward_cpu(float* dlogits, const float* dlosses, const float* probs, const int* targets, int B, int T, int V){
    // crossentropy of a batch
    // dlogits = B * T * V (must be zero-initialized, gradients are accumulated with +=)
    // dlosses = B * T
    // probs = B * T * V
    // targets = B * T (token ids)

    for(int b = 0; b < B; b++){
        for(int t = 0; t < T; t++){
            float* dlogits_bt = dlogits + b * T * V + t * V; // base address to save the gradient
            const float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            const int ix = targets[b * T + t];
            for(int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f; // simulating one hot vector
                // dloss is dL/dloss for this position, typically 1.0f / (B*T) for a mean loss
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ------------------------------------------------------------------------------
// gpu kernel code
__global__ void crossentropy_softmax_backward_kernel1(float* dlogits, 
                                                      const float* dlosses, 
                                                      const float* probs, const int* targets,
                                                      int B, int T, int V){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B * T * V) {
        int b = i / (T * V);
        int t = (i / v) % T;
        int v = i % v;
        float* dlogits_bt = dlogits + b * T * V + t * V;
        float* probs_bt = probs + b * T * V + t * V;
        float dloss = dlosses[b * T + t];
        int ix = targets[b * T + t];
        float p = probs_bt[v];
        float indicator = v == ix ? 1.0f : 0.0f;
        dlogits_bt[v] += (p - indicator) * dloss;
    }
}

// ------------------------------------------------------------------------------
// kernel launcher


// kernel version dispatch



// ------------------------------------------------------------------------------

int main(){
    printf("start\n");

    int B = 8;
    int T = 1024;
    int v = 50257;

    float* probs = make_random_float01(B * T * V);
    int* targets = make_radom_int(B * T, V);
    float* dlosses = make_random_float01(B * T);
    float* dlogits = make_zero_float(B * T * V);



    return 0;
}

