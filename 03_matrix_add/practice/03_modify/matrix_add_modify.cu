// 1. 头文件
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

// 2. CUDA错误检查宏
#define CUDA_CHECK(call)                                   \
    do {                                                    \
        cudaError_t error = (call);                         \
        if (error != cudaSuccess) {                         \
            std::cerr << "CUDA error: "                     \
                      << cudaGetErrorString(error)          \
                      << " at " << __FILE__                 \
                      << ':' << __LINE__                    \
                      << '\n';                              \
            std::exit(EXIT_FAILURE);                        \
        }                                                   \
    } while (0)

// 3. Matrix Add Kernel
__global__ void matrix_add(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height) {

// 3.1 计算全局列号
    int col = 
        blockIdx.x * blockDim.x + threadIdx.x;

// 3.2 计算全局行号
    int row = 
        blockIdx.y * blockDim.y + threadIdx.y;

// 3.3 检查行列边界
    if (row < height && col < width) {
        // 3.4 二维坐标转一维下标
        int idx = 
            row * width + col;
        
        // 3.5 完成对应元素加法
        c[idx] = 
            a[idx] + b[idx];
    }
}


// 4. main函数
int main() {
// 4.1 定义矩阵行数和列数
    const int height = 5;
    const int width = 7;

// 4.2 计算元素数量和字节数
    const int element_count =
        height * width;
    
    const std::size_t bytes = 
        static_cast<std::size_t>(element_count) *
        sizeof(float);

// 4.3 创建CPU输入输出矩阵
    std::vector<float> h_a(element_count);
    std::vector<float> h_b(element_count);
    std::vector<float> h_c(element_count);

// 4.4 初始化输入矩阵
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = 
                row * width + col;

            h_a[idx] = 
                static_cast<float>(
                    row * 10 + col
                );

            h_b[idx] = 
                static_cast<float>(100);
        }
    }

// 4.5 创建GPU指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

// 4.6 申请GPU显存
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

// 4.7 CPU输入复制到GPU
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

// 4.8 设置二维Block和Grid
    dim3 threads_per_block(3, 2);
    
    dim3 blocks(
        (width + threads_per_block.x - 1) /
            threads_per_block.x,

        (height + threads_per_block.y - 1) /
            threads_per_block.y
    );

    std::cout
    << "matrix height = "
    << height
    << '\n'
    << "matrix width = "
    << width
    << '\n'
    << "threads per block = (x="
    << threads_per_block.x
    << ", y="
    << threads_per_block.y
    << ")\n"
    << "blocks = (x="
    << blocks.x
    << ", y="
    << blocks.y
    << ")\n\n";

// 4.9 启动Kernel
    matrix_add<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        width,
        height
    );

// 4.10 检查并等待GPU
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

// 4.11 GPU结果复制回CPU
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

// 4.12 打印结果
    std::cout << "\nMatrix C:\n";

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = 
                row * width + col;

            std::cout
                << h_c[idx]
                << '\t';
        }
        std::cout << '\n';
    }

// 4.13 释放GPU显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

// 4.14 返回成功
    return EXIT_SUCCESS;
}
