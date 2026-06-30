#include<stdio.h>
#include<stdlib.h>


template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}


// -------------------------------------------------------------------
// Helper Methods

// cuda error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

}

// cudaCheck macro
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))


// -------------------------------------------------------------------
// testing and benchmarking utils 
template<class TargetType>
[[nodiscard]] cudaError_t memcpy_convert(TargetType* d_ptr, float* h_ptr, size_t count) {
    // copy from host to device with data type conversion
    TargetType* converted = (TargetType*)malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++){
        converted[i] = (TargetType) h_ptr[i];
    }
    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);
    return status;
}

template<class D, class T>
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elemnts, T tolerance=1e-4){
    // copy the data from gpu to cpus
    D* out_gpu = (D*)malloc(num_elements * sizeof(D));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BF16
    float epsilon = FLT_EPSILON;
#else
    float epsilon = 0.079;
#endif
    for (int i = 0; i < num_elements; i++) {
        // skip masked elements
        if (!isfinite(cpu_reference[i]))
            continue;
        // print the few comparisions
        if (i < 5) {
            printf("%f %f\n", cpu_reference[i], (T)out_gpu[i]);
        }

        // effective tolerance is based on expected rounding error (epsilon),
        // plus any specified tolerance
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;

        // ensure correctness for all elements
        if (fabs(cpu_reference[i] - (T)out_gpu[i]) > t_eff) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], 
                   (T)out_gpu[i]);
            nfaults++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }

    }

    if (nfaults >= 10) {
        free(out_gpu);
        exit(EXIT_FAILURE);
    }
    free(out_gpu);
}

// -------------------------------------------------------------------
// reduced/mixed precision utilities
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#elif defined(ENABLE_FP16)

typedef half floatX;
typedef half floatN;

#else

typedef float floatX;
typedef float floatN;

#endif

typedef Packed128<floatX> x128;





// -------------------------------------------------------------------
// random utils

float* make_random_float(size_t N) {
	float* arr = (float*)malloc(N * sizeof(float));

	for (size_t i = 0; i < N; i++) {
		arr[i] = ((float)rand() / RAND_MAX); // 0..1
	}
	return arr;
}
