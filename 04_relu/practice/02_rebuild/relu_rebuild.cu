// 1. 头文件
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

// 2. CUDA错误检查宏
#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t error = (call);                 \
        if (error != cudaSuccess) {                 \
            std::cerr << "CUDA error: "             \
                      << cudaGetErrorString(error)  \
                      << " at " << __FILE__         \
                      << ":" << __LINE__              \
                      << '\n';                      \
            std::exit(EXIT_FAILURE);                \
        }                                           \
    } while (0)


// 3. ReLU Kernel
__global__ void relu(
    const float* input,
    float* output,
    int n) {

// 3.1 计算全局线程下标
    int idx = 
        blockIdx.x * blockDim.x + threadIdx.x;

// 3.2 检查边界
    if (idx < n) {
    // 3.3 三目运算符实现ReLU
        output[idx] = 
            input[idx] > 0.0F
            ? input[idx]
            : 0.0F;
    }
}

// 4. main函数
int main() {

// 4.1 创建CPU输入
    std::vector<float> h_input ={
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
// 4.2 计算n和bytes
    const int n =
        static_cast<int>(h_input.size());

    const std::size_t bytes = 
        static_cast<std::size_t>(n) *
            sizeof(float);

// 4.3 创建CPU参考结果和GPU返回结果
    std::vector<float> h_cpu_output(n);

    std::vector<float> h_gpu_output(n);

// 4.4 CPU计算参考结果
    for (int idx = 0; idx < n; ++idx) {
        h_cpu_output[idx] = 
            h_input[idx] > 0.0F
            ? h_input[idx]
            : 0.0F;
    }

// 4.5 创建GPU指针
    float* d_input = nullptr;
    float* d_output = nullptr;

// 4.6 申请GPU显存
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        bytes
    ));

// 4.7 CPU输入复制到GPU
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

// 4.8 配置线程
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
        << "total threads = "
        << blocks * threads_per_block
        << "\n\n";

// 4.9 启动Kernel
    relu<<<blocks, threads_per_block>>>(
        d_input,
        d_output,
        n
    );

// 4.10 检查并等待GPU
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

// 4.11 GPU结果复制回CPU
    CUDA_CHECK(cudaMemcpy(
        h_gpu_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

// 4.12 比较CPU和GPU结果
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
            <<'\t'
            << (correct ? "OK" : "ERROR")
            << '\n';
    }

// 4.13 输出最终验证结果
    std::cout
        << "\nVerification: "
        << (all_correct ? "PASS" : "FAIL")
        <<'\n';

// 4.14 释放GPU显存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

// 4.15 返回结果
    return all_correct
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
