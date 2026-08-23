#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
do                                                            \
{                                                             \
    cudaError_t error = (call);                               \
                                                              \
    if(error != cudaSuccess)                                  \
    {                                                         \
        std::cerr                                             \
            << "CUDA Error: "                                 \
            << cudaGetErrorString(error)                       \
            << "\n";                                          \
                                                              \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
                                                              \
}while(0)

__inline__ __device__
float warp_reduce_sum(float value) {

    unsigned int mask = 0xffffffff;

    for (int offset = warpSize / 2;
        offset > 0;
        offset /= 2) {

        value += 
            __shfl_down_sync(
                mask,
                value,
                offset);
    }

    return value;
}

__global__ void reduce_kernel(
    const float* input,
    float* output,
    int n) {

    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (idx < n) {

        sum += input[idx];
    }

    if (idx + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }

    sum = warp_reduce_sum(sum);

    const int lane = tid % warpSize;

    const int warp_id = tid / warpSize;

    if (lane == 0) {
        
        shared[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0) {

        const int num_warps = blockDim.x / warpSize;

        if (lane < num_warps) {

            sum = shared[lane];
        } else {

            sum = 0.0F;
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

int main() {

    constexpr int n = 1 << 22;
    constexpr int BLOCK_SIZE = 256;

    int blocks = 
        (n + BLOCK_SIZE * 2 - 1) /
        (BLOCK_SIZE * 2);

    std::cout << "n = "
              << n
              << "\n"
              << "blocks = "
              << blocks
              << "\n";

    std::vector<float> h_input(n, 1.0f);

    float* d_input = nullptr;
    float* d_partial = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input,n * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice));

    reduce_kernel<<<
        blocks,
        BLOCK_SIZE,
        (BLOCK_SIZE / 32) * sizeof(float)>>>(
            d_input,
            d_partial,
            n);

    CUDA_CHECK(cudaDeviceSynchronize());

    int remaining = blocks;

    while (remaining > 1) {

        int new_blocks =

            (remaining + BLOCK_SIZE* 2 - 1) /
            (BLOCK_SIZE * 2);

        reduce_kernel<<<
            new_blocks,
            BLOCK_SIZE,
            (BLOCK_SIZE / 32) * sizeof(float)>>>(
                d_partial,
                d_partial,
                remaining);

        remaining = new_blocks;
    }

    float gpu_result;

    CUDA_CHECK(cudaMemcpy(
            &gpu_result,
            d_partial,
            sizeof(float),
            cudaMemcpyDeviceToHost));

    float cpu_result = 0;

    for (float x : h_input) {

        cpu_result += x;
    }

    std::cout << "CPU = "
              << cpu_result
              << "\n"
              << "GPU = "
              << gpu_result
              << "\n"
              << "correct = "
              << std::boolalpha
              << (fabs(cpu_result - gpu_result) < 1e-5)
              << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}

