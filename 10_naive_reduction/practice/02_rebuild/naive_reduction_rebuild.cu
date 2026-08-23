#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t error = (call);                 \
        if (error != cudaSuccess) {                 \
            std::cerr << "CUDA error:"              \
                      << cudaGetErrorString(error)  \
                      << " at " << __FILE__         \
                      << ":" << __LINE__            \
                      << '\n';                      \
            std::exit(EXIT_FAILURE);                \
        }                                           \
    } while (0)

__global__ void reduce_naive(
    float* data,
    float* output,
    int n) {

    unsigned int tid = threadIdx.x;

    for (unsigned int stride = 1;
        stride < blockDim.x;
        stride *= 2) {

        if (tid % (2 * stride) == 0) {

            unsigned int other = tid + stride;
            
            if (tid < n && other < n) {
                data[tid] += data[other];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[0] = data[0];
    }
}

float cpu_reduce(const std::vector<float>& input) {
    float sum = 0.0F;

    for (float value : input) {
        sum += value;
    }

    return sum;
}

int main() {
    constexpr int N = 8;
    constexpr int BLOCK_SIZE = 8;

    std::vector<float> h_input(N);

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    float cpu_sum = cpu_reduce(h_input);

    float* d_data = nullptr;
    float* d_output = nullptr;

    size_t bytes = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
            d_data,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice));

    reduce_naive<<<1,BLOCK_SIZE>>>(
        d_data,
        d_output,
        N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_sum = 0.0F;
    
    CUDA_CHECK(cudaMemcpy(
            &gpu_sum,
            d_output,
            sizeof(float),
            cudaMemcpyDeviceToHost));

    bool correct =
        std::fabs(cpu_sum - gpu_sum) < 1e-5f;

    std::cout << "input: ";

    for (float value : h_input) {
        std::cout << value << " ";
    }

    std::cout << "\n";

    std::cout << "CPU sum = "
              << cpu_sum << "\n";

    std::cout << "GPU sum = "
              << gpu_sum << "\n";

    std::cout << std::boolalpha
              << "correct = "
              << correct << "\n";

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
