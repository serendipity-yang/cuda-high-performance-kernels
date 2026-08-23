#include <cuda_runtime.h>

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

// GPU Kernel：一个线程处理矩阵中的一个元素
__global__ void matrix_add(
    const float* a,
    const float* b,
    float* c,
    int width,
    int height) {

    // x 方向表示列
    int col =
        blockIdx.x * blockDim.x + threadIdx.x;

    // y 方向表示行
    int row =
        blockIdx.y * blockDim.y + threadIdx.y;

    printf(
        "block=(%d,%d), thread=(%d,%d), "
        "row=%d, col=%d\n",
        blockIdx.x,
        blockIdx.y,
        threadIdx.x,
        threadIdx.y,
        row,
        col
    );

    // 防止多余线程越界
    if (row < height && col < width) {
        // 二维坐标转换为一维数组下标
        int idx = row * width + col;

        c[idx] = a[idx] + b[idx];
    }
}

// 打印矩阵
void print_matrix(
    const std::vector<float>& matrix,
    int width,
    int height,
    const char* name) {

    std::cout << name << ":\n";

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = row * width + col;

            std::cout << matrix[idx] << '\t';
        }

        std::cout << '\n';
    }

    std::cout << '\n';
}

int main() {
    // 矩阵宽度：每行5个元素
    const int width = 7;

    // 矩阵高度：一共3行
    const int height = 5;

    // 矩阵总元素数量
    const int element_count = width * height;

    // 整个矩阵占用的字节数量
    const std::size_t bytes =
        static_cast<std::size_t>(element_count) *
        sizeof(float);

    // CPU 内存中的矩阵
    std::vector<float> h_a(element_count);
    std::vector<float> h_b(element_count);
    std::vector<float> h_c(element_count);

    // 初始化矩阵
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = row * width + col;

            h_a[idx] = static_cast<float>(idx);
            h_b[idx] = static_cast<float>(idx * 10);
        }
    }

    // GPU 显存指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // 在 GPU 中申请显存
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

    // 将 A、B 从 CPU 复制到 GPU
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

    // 每个线程块是二维的：
    // x方向4个线程，y方向2个线程
    dim3 block(4, 2);

    // 计算二维 Grid 大小
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    std::cout << "Matrix width  = "
              << width << '\n';

    std::cout << "Matrix height = "
              << height << '\n';

    std::cout << "Block shape   = ("
              << block.x << ", "
              << block.y << ")\n";

    std::cout << "Grid shape    = ("
              << grid.x << ", "
              << grid.y << ")\n";

    std::cout << "Total threads = "
              << grid.x * grid.y *
                 block.x * block.y
              << "\n\n";

    // 启动二维 CUDA Kernel
    matrix_add<<<grid, block>>>(
        d_a,
        d_b,
        d_c,
        width,
        height
    );

    // 检查 Kernel 启动错误
    CUDA_CHECK(cudaGetLastError());

    // 等待 GPU 执行完成
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果从 GPU 复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 打印三个矩阵
    print_matrix(h_a, width, height, "Matrix A");
    print_matrix(h_b, width, height, "Matrix B");
    print_matrix(h_c, width, height, "Matrix C");

    // 释放显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
