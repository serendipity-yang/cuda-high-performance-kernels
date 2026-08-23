#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t error = (call);                         \
        if (error != cudaSuccess) {                         \
            std::cerr << "CUDA error: "                     \
                      << cudaGetErrorString(error)          \
                      << " at " << __FILE__                 \
                      << ":" << __LINE__                    \
                      << '\n';                              \
            std::exit(EXIT_FAILURE);                        \
        }                                                   \
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
    std::vector<float> h_input = {
        -3.0F,
        -1.5F,
         0.0F,
         2.0F,
         5.5F,
        -7.0F,
         1.0F,
        -0.25F,
         8.0F,
        -2.0F
    };

    const int n = 
        static_cast<int>(h_input.size());

    const std::size_t bytes = 
        static_cast<std::size_t>(n) * 
        sizeof(float);

    std::vector<float> h_cpu_output(n);
    std::vector<float> h_gpu_output(n);

    for (int idx = 0; idx < n; ++idx) {
        h_cpu_output[idx] = 
            h_input[idx] > 0.0F 
            ? h_input[idx]
            : 0.0F;
    }

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

    const int threads_per_block = 4;

    const int blocks = 
        (n + threads_per_block - 1) /
        threads_per_block;
        
    std::cout
        << "element count = "
        << n
        << '\n'
        << "threads per block = "
        << threads_per_block
        << '\n'
        << "blocks = "
        << blocks
        << '\n'
        << "total threads = "
        << blocks * threads_per_block
        << "\n\n";

    relu<<<blocks, threads_per_block>>>(
        d_input,
        d_output,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_gpu_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));
    
    const float tolerance = 1.0e-6F;

    bool all_correct = true;

    std::cout 
        << "idx\tinput\tCPU\tGPU\tresult\n";

    for (int idx = 0; idx < n; ++idx) {
        float difference = 
            std::fabs(
                h_cpu_output[idx] - 
                h_gpu_output[idx]
            );
        bool correct = 
            difference <= tolerance;

        if (!correct) {
            all_correct = false;
        }

        std::cout
            << idx
            << '\t'
            << h_input[idx]
            << '\t'
            << h_cpu_output[idx]
            << '\t'
            << h_gpu_output[idx]
            << '\t'
            << (correct ? "OK" : "ERROR")
            << '\n';

    }
    std::cout 
        << "\nVerification: "
        << (all_correct ? "PASS" : "FAIL")
        << '\n';
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return all_correct 
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
