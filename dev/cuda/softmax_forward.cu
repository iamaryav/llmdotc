#include <assert.h>
#include "common.h"

//---------------------------------------------------------------------------------------
// CPU code ref

void softmax_forward_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        // calculate max for each row
        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        // denominator - sum over all the logits
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        // softmax calculation for word in vocab for each token
        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

// Online normalizer calculation for softmax
void softmax_forward_online_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        // online max/sum calculation
        float max_val = -INFINITY;
        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            float max_val_prev = max_val;
            if (max_val < inp_row[j]) {
                max_val = inp_row[j];
                sum = sum * expf(max_val_prev - max_val) + expf(inp_row[j] - max_val);
            } else {
                sum += expf(inp_row[j] - max_val);
            }
        }

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - max_val) / (float)sum;
        }
    }
}

//---------------------------------------------------------------------------------------
// GPU Kernels




//---------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    int B = 8;
    int T = 64;
    int V = 50257;
    int N = B * T * V;

    float* out = (float*)malloc(N * sizeof(float));
    float* out2 = (float*)malloc(N * sizeof(float));
    float* inp = make_random_float(N);

    int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k){
        for (int j = 0; j < B * T; ++j) {
            inp[j * V + outliers[j*3 + k]] *= 20;
        }
    }
    softmax_forward_cpu(out, inp, B*T, V);
    {
        float max_el = -INFINITY;
        for (int i = 0; i < B * T * V; ++i) {
            max_el = max(max_el, out[i]);
        }
        assert(max_el > 1e-4);
        printf("Largest output is: %f\n", max_el);
    }

    softmax_forward_online_cpu(out2, inp, B*T, V);
    {
        // online softmax must match the 3-pass reference
        float max_diff = 0.0f;
        for (int i = 0; i < B * T * V; ++i) {
            max_diff = max(max_diff, fabsf(out[i] - out2[i]));
        }
        printf("Online vs reference max abs diff: %e\n", max_diff);
        assert(max_diff < 1e-6);
    }


    free(inp);
    free(out);
    free(out2);
    free(outliers);
    return 0;

    
}
