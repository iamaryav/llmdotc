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
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C) {
    // thread per row naive way
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float max_val = -INFINITY;
    if (idx < N){
        const float* inp_row = inp + idx * C;
        float* out_row = out + idx * C;

        for (int j = 0; j < C; j++) {
            if (max_val < inp_row[j]) {
                max_val = inp_row[j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - max_val);
            sum += out_row[j];
        }

        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

__global__ void softmax_forward_kernel2(float* out, const float* inp, int N, int C) {
    // one block per row
    // thread coarsening, parallel reduction, syncthreads
    extern __shared__ float shared[];
    int idx = blockIdx.x; // block index
    int tid = threadIdx.x;
    const float* x = inp + idx * C; // idx-the row in the inp
    int block_size = blockDim.x;
    float max_val = -INFINITY;
    // thread coarsening
    for(int i = tid; i < C; i += block_size){
            max_val = fmaxf(max_val, x[i]);
    }
    shared[tid] = max_val;

    // parallel reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride){
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }

    __syncthreads();
    float offset = shared[0];
    // exponent calculation
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    __syncthreads();
    // sum using thread coarsening
    x = out + idx * C; // idx-th row of out
    float sum_val = 0.0f;
    for(int i = tid; i < C; i += block_size) {
        sum_val += x[i];
    }
    shared[tid] = sum_val;

    //reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}

__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C) {
    // one warp per row
}

//---------------------------------------------------------------------------------------

// kernel launcher
void softmax_forward1(float* out, const float* inp, int N, int C, const int block_size) {
    // one thread per row
    int grid_size = ceil_div(N, block_size)
    softmax_forward_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward2(float* out, const float* inp, int N, int C, const int block_size) {
    // one block per row, one shared float per thread
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel2<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
}

// kernel version dispatch
void softmax_forward(int kernel_num, float* out, const float* inp, int N, int C, int block_size) {
    switch (kernel_num) {
        case 1:
            softmax_forward1(out, inp, N, C, block_size);
            break;
        case 2:
            softmax_forward2(out, inp, N, C, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}




// nsighe compute 
//---------------------------------------------------------------------------------------
int main(int argc, char** argv) {
    int B = 8;
    int T = 64;
    int V = 50257;
    int N = B * T * V;

    // read the kernel from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    float* out = (float*)malloc(N * sizeof(float));
    float* out2 = (float*)malloc(N * sizeof(float));
    float* inp = make_random_float(N);

    int* outliers = make_random_int(B * T * 3, V);
    for(int k = 0; k < 3; ++k){
        for (int j = 0; j < B * T; ++j) {
            inp[j * V + outliers[j*3 + k]] *= 20;
        }
    }

    // device
    float* d_out;
    float* d_inp;

    // allocate memory on the device
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, N * sizeof(float)));

    // move input from host to device
    cudaCheck(cudaMemcpy(d_inp, inp, N * sizeof(float), cudaMemcpyHostToDevice));

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

    // check the kernel against the CPU reference at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_forward(kernel_num, d_out, d_inp, B * T, V, block_size);
        validate_result(d_out, out, "out", N, 1e-4f);
    }
    printf("All results matched. Starting benchmarks.\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        int repeat_times = 100;

        float elapsed_time = benchmark_kernel(repeat_times, softmax_forward,
                                              kernel_num, d_out, d_inp,
                                              B * T, V, block_size);

        // napkin math: read inp twice (max, exp), write out, then read + write out (normalize)
        // kernel 2 does 6 passes
        long memory_ops = (long)N * 5 * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }




    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    free(inp);
    free(out);
    free(out2);
    free(outliers);
    return 0;

    
}
