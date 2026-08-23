#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// 检查 CUDA API 是否执行成功
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

// GPU Kernel
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n) {

    // 计算当前线程的全局编号
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

     printf(
        "block=%d, thread=%d, idx=%d\n",
        blockIdx.x,
        threadIdx.x,
        idx
    );

    // 防止多余线程访问数组外部
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // 向量中有 10 个元素
    const int n = 17;

    // 10 个 float 总共占用的字节数
    const std::size_t bytes =
        n * sizeof(float);

    // CPU 内存中的四个向量
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    // 初始化输入数据
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 10);
    }

    // GPU 显存指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // 在 GPU 显存中申请空间
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

    // 将输入数据从 CPU 复制到 GPU
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

    // 每个 Block 使用 4 个线程
    const int threads_per_block = 4;

    // 计算需要多少个 Block
    const int blocks =
        (n + threads_per_block - 1) /
        threads_per_block;

    std::cout << "n = " << n << '\n';
    std::cout << "threads_per_block = "
              << threads_per_block << '\n';
    std::cout << "blocks = "
              << blocks << '\n';
    std::cout << "total threads = "
              << blocks * threads_per_block
              << "\n\n";

    // 启动 GPU Kernel
    vector_add<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        n
     );

    // 检查 Kernel 启动错误
    CUDA_CHECK(cudaGetLastError());

    // 等待 GPU 执行完毕
    CUDA_CHECK(cudaDeviceSynchronize());

    // 把计算结果从 GPU 复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 输出结果
    for (int i = 0; i < n; ++i) {
        std::cout
            << h_a[i]
            << " + "
            << h_b[i]
            << " = "
            << h_c[i]
            << '\n';
    }

    // 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
