#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = (call);                                     \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA error: "                                 \
                      << cudaGetErrorString(error)                       \
                      << " at " << __FILE__                              \
                      << ':' << __LINE__                                 \
                      << '\n';                                          \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
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
    // CPU输入数组：包含负数、0和正数
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

    // 元素总数
    const int n =
        static_cast<int>(h_input.size());

    // 数组占用的总字节数
    const std::size_t bytes =
        static_cast<std::size_t>(n) *
        sizeof(float);

    // CPU输出数组
    std::vector<float> h_output(n);

    // GPU显存指针
    float* d_input = nullptr;
    float* d_output = nullptr;

        // 为GPU输入数组申请显存
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        bytes
    ));

    // 为GPU输出数组申请显存
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        bytes
    ));

    // CPU输入复制到GPU
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

        
    // 每个Block有4个线程
    const int threads_per_block = 4;

    // 向上取整计算Block数量
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
    // 启动ReLU Kernel
    relu<<<blocks, threads_per_block>>>(
        d_input,
        d_output,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU输出复制回CPU
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    std::cout
        << "idx\tinput\toutput\n";

    for (int idx = 0; idx < n; ++idx) {
        std::cout
            << idx
            << '\t'
            << h_input[idx]
            << '\t'
            << h_output[idx]
            << '\n';
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return EXIT_SUCCESS;
}
