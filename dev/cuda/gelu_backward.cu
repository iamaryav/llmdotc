#include<stdio.h>
#include<stdlib.h>








//-------------------------------------------------------------
// CPU code ref

#define GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI);

void gelu_backward_cpu(float* dinp, const float* inp, const float* dout, const int N){
    // approximation formula 
    for (int i = 0; i < N; i++) {
        x = inp[i]
        float cube = 0.044715f * x * x * x;
        float tanh_args = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_args);
        float cosh_out = coshf(tanh_args);
        float sech_out = 1.0f / (cosh_out * cosh_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + 0.5f * x * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = (float)(local_grad * (float)dout[i]);
    }

}


//-------------------------------------------------------------

int main(int argc, char** argv) {
    // setup main

    int B = 8;
    int T = 1024;
    int C = 768;

    float* d
    return 0
}
