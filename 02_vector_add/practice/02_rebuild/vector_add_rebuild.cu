// 1. 引入头文件
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = (call);                                   \
        if (error != cudaSuccess) {                                   \
            std::cerr << "CUDA error: "                               \
                      << cudaGetErrorString(error)                     \
                      << " at line " << __LINE__ << std::endl;         \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

// 2. 编写 vector_add Kernel
__global__ void vector_add (
    const float* a,
    const float* b,
    float* c,
    int n) {

    int idx = 
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 3. 创建 CPU 输入输出数组
int main() {
    const int n = 10;

    const std::size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

// 4. 初始化 CPU 输入
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 10);
    }

// 5. 创建 GPU 指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

// 6. 申请 GPU 显存
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_a),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_b),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_c),
        bytes
    ));

// 7. 将输入复制到 GPU
    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

// 8. 计算 blocks 和 threads
    const int threads_per_block = 4;
    const int blocks = 
        (n + threads_per_block - 1) /
        threads_per_block;

// 9. 启动 Kernel
    vector_add<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        n
    );

// 10. 等待 GPU 完成
    CUDA_CHECK(cudaDeviceSynchronize());

// 11. 将结果复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
        ));

// 12. 打印或检查结果
    for (int i = 0; i < n; ++i) {
        std::cout
            << h_a[i]
            << " + "
            << h_b[i]
            << " = "
            << h_c[i]
            << '\n';
    }


// 13. 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
