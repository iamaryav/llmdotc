#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<math.h>


template<class T>
__host__ __device__ T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// -------------------------------------------------------------------
// Packed128 data structure, which forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instruction)
template<class ElementType>
struct alignas(16) Packed128{
    Packed128() = default;
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__ static Packed128 constant(ElementType value) {
        Packed128 result;
        for (int k = 0; k < size; k++){
            result.payload[k] = value;
        }
        return result;
    }

    __device__ static Packed128 zeros(){
        return constant(0);
    }

    __device__ static Packed128 ones(){
        return constant(1);
    }

    __device__ ElementType& operator[](int index) {
        return payload[index];
    }

    __device__ int4 get_bits() {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    // e.g. sizof(int4) is 16 (4 x 4 bytes), sizeof(bfloat16) =2, so size = 8
    // so in the case where ElementType = bfloat16, we store 8 elements in on packed 128
    static constexpr const int size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};

// short-form typedefs
typedef Packed128<float> f128;

// load a packed128 from a aligned memory address
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

// load a Packed128 from an aligned memory address with streaming cache hint
// means don't cache this data skip caching read once data
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}

// store a Packed128 to an aligned memory address
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}


// store a Packed128 to an aligned memory address with streaming cache hint
// write the data but don't store in cache
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(*reinterpret_cast<int4*>(target) = value.get_bits());
}

// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(*reinterpret_cast<int4*>(target) = value.get_bits());
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
void validate_result(D* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance=1e-4){
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

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    // scrub l2 cache between benchmakrs

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    float elapsed_time = 0.f;
    for (int i = 0; i < repeats; i++){
        // clear L2
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        // start recording the timing of the kernel
        cudaCheck(cudaEventRecord(start, nullptr));
        kernel(std::forward<KernelArgs>(kernel_args)...);
        cudaCheck(cudaEventRecord(stop, nullptr));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float single_call;
        cudaCheck(cudaEventElapsedTime(&single_call, start, stop));
        elapsed_time += single_call;
    }
    cudaCheck(cudaFree(flush_buffer));
    return elapsed_time / repeats;
}



// -------------------------------------------------------------------
// reduced/mixed precision utilities
#if defined(ENABLE_BF16)
typedef __nv_bfloat16 floatX;
typedef __nv_bfloat16 floatN;
#define CUBLAS_LOWP CUDA_R_16BF
#define CUBLAS_LOWP_COMPUTE CUBLAS_COMPUTE_32F

#elif defined(ENABLE_FP16)

#include <cuda_fp16.h>
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
