#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                           \
    do {                                           \
        cudaError_t error = (call);                \
        if (error != cudaSuccess) {                \
            std::cerr                              \
                << "CUDA error: "                  \
                << cudaGetErrorString(error)       \
                << " at "                          \
                << __FILE__                        \
                << ":"                             \
                << __LINE__                        \
                << '\n';                           \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)

__global__ void reduce_interleaved(
    const float* input,
    float* partial_sum,
    int n) {

    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;

    const unsigned int global_idx = blockIdx.x * blockDim.x * 2 + tid;

    float value = 0.0F;

    if (global_idx < n) {

        value += input[global_idx];
    }

    const unsigned int second_idx =
        global_idx + blockDim.x;

    if (second_idx < n) {

        value += input[second_idx];
    }

    sdata[tid] = value;

    __syncthreads();

    for (unsigned int stride = 1;
        stride < blockDim.x;
        stride *= 2) {

        if (tid % (2 * stride) == 0) {

            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {

        partial_sum[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sequential(
    const float* input,
    float* partial_sum,
    int n) {

    extern __shared__ float sdata[];

    const unsigned int tid = threadIdx.x;

    const unsigned int global_idx = blockIdx.x * blockDim.x * 2 + tid;

    float value = 0.0F;

    if (global_idx < n) {

        value += input[global_idx];
    }

    const unsigned int second_idx = global_idx + blockDim.x;


    if (second_idx < n) {

    
        value += input[second_idx];
    }
    
    sdata[tid] = value;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1) {

        if (tid < stride) {

            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();

    }

    if (tid == 0) {
    
    partial_sum[blockIdx.x] = sdata[0];
    }
}

double correct_partial_sum(const std::vector<float>& partial) {

    double sum = 0.0;

    for (float value : partial) {

        sum += static_cast<double>(value);
    }

    return sum;
}

float benchmark_interleaved(
    const float* d_input,
    float* d_partial,
    int n,
    int blocks,
    int threads,
    int repeat) {

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const std::size_t shared_bytes = threads * sizeof(float);

    for (int i = 0; i < 10; ++i) {

        reduce_interleaved<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {

        reduce_interleaved<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0F;

    CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(repeat);
}

float benchmark_sequential(
    const float* d_input,
    float* d_partial,
    int n,
    int blocks,
    int threads,
    int repeat) {

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::size_t shared_bytes = threads * sizeof(float);

    for (int i = 0; i < 10; ++i) {
        reduce_sequential<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeat; ++i) {

        reduce_sequential<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0F;

    CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return elapsed_ms / static_cast<float>(repeat);
}

int main() {

    constexpr int n = 1 << 22;

    constexpr int threads = 256;
    
    constexpr int repeat = 100;

    const int blocks = 
        (n + threads * 2 - 1) / threads * 2;

    std::cout << " n    = "
              << n
              << '\n';

    std::cout << "threads = "
              << threads
              << '\n';

    std::cout << "blocks = "
              << blocks
              << '\n';

    std::vector<float> h_input(n, 1.0F);

    double cpu_sum = 0.0;

    for (float value : h_input) {

        cpu_sum += static_cast<double>(value);
    }

    float* d_input = nullptr;
    float* d_partial = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

    const std::size_t shared_bytes = threads * sizeof(float);

    reduce_interleaved<<<
        blocks,
        threads,
        shared_bytes>>>(
            d_input,
            d_partial,
            n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partial_interleaved(blocks);

    CUDA_CHECK(cudaMemcpy(
            h_partial_interleaved.data(),
            d_partial,
            blocks * sizeof(float),
            cudaMemcpyDeviceToHost));

    const double interleaved_sum =
        correct_partial_sum(h_partial_interleaved);


    reduce_sequential<<<
        blocks,
        threads,
        shared_bytes>>>(
            d_input,
            d_partial,
            n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_partial_sequential(blocks);

    CUDA_CHECK(cudaMemcpy(
            h_partial_sequential.data(),
            d_partial,
            blocks * sizeof(float),
            cudaMemcpyDeviceToHost));

    const double sequential_sum =
        correct_partial_sum(h_partial_sequential);

    const float interleaved_ms = 
        benchmark_interleaved(
            d_input,
            d_partial,
            n,
            blocks,
            threads,
            repeat);

    const float sequential_ms =
        benchmark_sequential(
            d_input,
            d_partial,
            n,
            blocks,
            threads,
            repeat);

    std::cout << "\nCPU sum     = "
              << cpu_sum
              << '\n';

    std::cout << "interleaved sum = "
              << interleaved_sum
              << '\n';
    
    std::cout << "sequential sum = "
              << sequential_sum
              << '\n';

    const bool interleaved_correct =
        std::fabs(interleaved_sum - cpu_sum) < 1e-3;

    const bool sequential_correct =
        std::fabs(sequential_sum - cpu_sum) < 1e-3;

    std::cout << std::boolalpha;

    std::cout << "\ninterleaved correct = "
              << interleaved_correct
              << '\n';

    std::cout << "sequential correct = "
              << sequential_correct
              << '\n';

    std::cout << "\ninterleaved average = "
              << interleaved_ms
              << "ms\n";

    std::cout << "sequential average = "
              << sequential_ms
              << "ms\n";

    std::cout << "interleaved / sequential = "
              << interleaved_ms / sequential_ms
              << "x\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
