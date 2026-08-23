#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t error = (call);                     \
        if (error != cudaSuccess) {                     \
            std::cerr << "CUDA error: "                 \
                      << cudaGetErrorString(error)      \
                      << " at " << __FILE__             \
                      << ":" << __LINE__                \
                      << '\n';                          \
            std::exit(EXIT_FAILURE);                    \
        }                                               \
    } while (0)

__global__ void relu(
    const float* input,
    float* output,
    int n) {

    int idx = 
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        output[idx] = 
            input[idx] > 0.0F
                ? input[idx]
                : 0.0F;
    }
}

int main() {
    const int n = 10'000'000;

    const std::size_t bytes = 
        static_cast<std::size_t>(n) * 
        sizeof(float);

    std::vector<float> h_input(n);
    std::vector<float> h_cpu_output(n);
    std::vector<float> h_gpu_output(n);

    for (int idx = 0; idx < n; ++idx) {
        const float value = 
            static_cast<float>(idx % 1000) /
            10.0F;

        h_input[idx] = 
            idx % 2 == 0
                ? value
                : -value;
    }

    const auto cpu_start = 
        std::chrono::steady_clock::now();

    for (int idx = 0; idx < n; ++idx) {
        h_cpu_output[idx] = 
            h_input[idx] > 0.0F
                ? h_input[idx]
                : 0.0F;
    }

    const auto cpu_stop = 
        std::chrono::steady_clock::now();

    const double cpu_ms = 
        std::chrono::duration<
            double,
            std::milli
        >(
            cpu_stop - cpu_start
        ).count();

    std::cout 
        << "element count = "
        << n
        << '\n'
        << "data size = "
        << static_cast<double>(bytes) /
                (1024.0 * 1024.0)
        << " MiB\n"
        << "CPU ReLU time = "
        << cpu_ms
        << " ms\n";

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        bytes
    ));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    const int threads_per_block = 512;

    const int blocks = 
        (n + threads_per_block - 1) /
        threads_per_block;

    const int warmup_iterations = 0;

    for (int iteration = 0; 
        iteration < warmup_iterations;
        ++iteration) {

        relu<<<blocks, threads_per_block>>>(
            d_input,
            d_output,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    CUDA_CHECK(cudaEventCreate(&gpu_start));

    CUDA_CHECK(cudaEventCreate(&gpu_stop));

    const int benchmark_iterations = 100;

    CUDA_CHECK(cudaEventRecord(gpu_start));

    for (int iteration = 0; 
        iteration < benchmark_iterations;
        ++iteration) {

        relu<<<blocks, threads_per_block>>>(
            d_input,
            d_output,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(gpu_stop));

    CUDA_CHECK(cudaEventSynchronize(gpu_stop));
    
    float total_gpu_ms = 0.0F;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_gpu_ms,
        gpu_start,
        gpu_stop
    ));

    const float average_gpu_ms = 
        total_gpu_ms /
        static_cast<float>(benchmark_iterations);

    CUDA_CHECK(cudaMemcpy(
        h_gpu_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    const float tolerance = 1.0e-6F;
    bool all_correct = true;

    for (int idx = 0; idx < n; ++idx) {
        const float difference = 
            std::fabs(
                h_cpu_output[idx] -
                h_gpu_output[idx]
            );

        if (difference > tolerance) {
            all_correct = false;

            std::cerr
                << "Mismatch at idx = "
                << idx
                << ", CPU = "
                << h_cpu_output[idx]
                << ", GPU = "
                << h_gpu_output[idx]
                << '\n';

            break;
        }
    }

    std::cout 
        << "GPU total time for "
        << benchmark_iterations
        << "iterations = "
        << total_gpu_ms
        << "ms\n"
        << "GPU average kernel time = "
        << average_gpu_ms
        << " ms\n"
        << "Verification = "
        << (all_correct ? "PASS" : "FAIL")
        << '\n';

    if (average_gpu_ms > 0.0F) {
        const double speed_ratio = 
            cpu_ms /
            static_cast<double>(average_gpu_ms);

        std::cout 
            << "CPU time / GPU average kernel time = "
            << speed_ratio
            << '\n';
    }

    CUDA_CHECK(cudaEventDestroy(gpu_start));

    CUDA_CHECK(cudaEventDestroy(gpu_stop));

    CUDA_CHECK(cudaFree(d_input));

    CUDA_CHECK(cudaFree(d_output));

    return all_correct
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
