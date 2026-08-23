#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = (call);                                     \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA error: "                                 \
                      << cudaGetErrorString(error)                      \
                      << " at line " << __LINE__                        \
                      << '\n';                                          \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

__global__ void matrix_add(
    const float* a,
    const float* b,
    float* c,
    int width,        //矩阵列数
    int height) {     //矩阵行数

    // x方向对应矩阵的列
    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

     // y方向对应矩阵的行
    int row =
        blockIdx.y * blockDim.y + threadIdx.y;
    
    // 暂时观察二维线程编号
    printf(
    "block=(%u,%u), thread=(%u,%u), row=%d, col=%d\n",
    blockIdx.x,
    blockIdx.y,
    threadIdx.x,
    threadIdx.y,
    row,
    col
);
    // 行和列都不能超出矩阵范围
    if(row < height && col < width) {
        // 将二维坐标转换成一维连续内存下标
        int idx = 
            row * width + col;
    
    c[idx] =
        a[idx] + b[idx];
    }
}

int main() {
    //矩阵有3行、5列
    const int height = 3;
    const int width = 5;
    
    // 矩阵元素总数
    const int element_count = 
        height * width;

    // 一个矩阵占用的总字节数
    const std::size_t bytes = 
        static_cast<std::size_t>(element_count) * sizeof(float);

    // CPU内存中的两个输入矩阵和一个输出矩阵
    std::vector<float> h_a(element_count);
    std::vector<float> h_b(element_count);
    std::vector<float> h_c(element_count);

    // 给CPU输入矩阵赋值
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // 将二维坐标转换成一维下标
            int idx = 
                row * width + col;

            h_a[idx] = 
                static_cast<float> (idx);

            h_b[idx] = 
                static_cast<float> (idx);

        }
    }

    // GPU显存指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // 为三个矩阵申请GPU显存
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

    // 把输入矩阵A从CPU复制到GPU
    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 把输入矩阵B从CPU复制到GPU
    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 二维线程配置
    dim3 threads_per_block(2, 2);

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
        << "threads per block = ("
        << threads_per_block.x
        << ", "
        << threads_per_block.y
        << ")\n"
        << "blocks = ("
        << blocks.x
        << ", "
        << blocks.y
        << ")\n\n";

    // 启动Kernel
    matrix_add<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        width,
        height
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU结果复制回CPU
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 按矩阵形式打印结果
    std::cout << "\nMatrix C:\n";

    for (int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            int idx = 
                row * width + col;

            std::cout 
                << h_c[idx]
                << '\t';
        }

        std::cout << '\n';
    }

    // 打印详细结果
    std::cout << "\nDetailed reasults:\n";

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++ col) {
            int idx = 
                row * width + col;

            std::cout 
                << "("
                << row
                << ","
                << col
                << ") "
                << h_a[idx]
                << " + "
                << h_b[idx]
                << " = "
                << h_c[idx]
                << '\n';
        }
    }

    // 释放GPU显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}

