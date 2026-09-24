#include <stdio.h>
#include <stdlib.h>

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

int* make_random_int(size_t size, int v){
    int* val = (int*)malloc(size * sizeof(int));
    for (size_t i = 0; i < size; i++){
        val[i] = rand() % v;
    }
    return val;
}

float* make_random_float(size_t N){
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++){
        // range = -1..1
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0; 
    }
    return arr;
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

    encoder_forward_cpu(out, inp, wte, wpe, B, T, C);

    for(int i = 0; i <= 5; i++){
        printf("inp %d, wte %f, wpe %f, out %f\n",inp[i], wte[i], wpe[i], out[i]);
    }

    // verify: out[i] should equal wte[inp[0]] + wpe[0] for the first token
    int t = 0, ix = inp[0];
    for (int i = 0; i < 5; i++){
        printf("%f vs %f\n", out[i], wte[ix * C + i] + wpe[t * C + i]);
    }

    free(out);
    free(inp);
    free(wte);
    free(wpe);
    return 0;
}
