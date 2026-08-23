#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = (call); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " \
                      << cudaGetErrorString(error) \
                      << " at " << __FILE__ \
                      << ':' << __LINE__ \
                      << '\n'; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vector_add(
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

int main() {
    // 1. 修改数据规模
    const int n = 1003;

    const std::size_t bytes =
        static_cast<std::size_t>(n) * sizeof(float);

    // 2. 创建 CPU 输入输出数组
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    // 3. 修改输入数据
    for (int i = 0; i < n; ++i) {
        const float value =
            static_cast<float>(i);

        h_a[i] = 0.5F * value;
        h_b[i] = 2.0F * value + 1.0F;
    }

    // 4. 创建 GPU 指针
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // 5. 申请 GPU 显存
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

    // 6. 将输入复制到 GPU
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

    // 7. 修改线程配置
    const int threads_per_block = 256;

    const int blocks =
        (n + threads_per_block - 1) /
        threads_per_block;

    std::cout
        << "n = " << n
        << '\n'
        << "threads per block = "
        << threads_per_block
        << '\n'
        << "blocks = " << blocks
        << '\n'
        << "total threads = "
        << blocks * threads_per_block
        << "\n\n";

    // 8. 启动 Kernel
    vector_add<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        n
    );

    // 9. 检查并等待 GPU 完成
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 10. 将结果复制回 CPU
    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // 11. 检查全部结果，只打印部分结果
    bool correct = true;

    for (int i = 0; i < n; ++i) {
        const float expected =
            h_a[i] + h_b[i];

        const float error =
            std::fabs(h_c[i] - expected);

        if (error > 1e-5F) {
            std::cerr
                << "Mismatch at index " << i
                << ": expected " << expected
                << ", got " << h_c[i]
                << ", error " << error
                << '\n';

            correct = false;
            break;
        }

        if (i < 5 || i >= n - 5) {
            std::cout
                << "index " << i
                << ": "
                << h_a[i]
                << " + "
                << h_b[i]
                << " = "
                << h_c[i]
                << '\n';
        }

        if (i == 5) {
            std::cout << "...\n";
        }
    }

    if (correct) {
        std::cout
            << "\nAll "
            << n
            << " results are correct.\n";
    }

    // 12. 释放 GPU 显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    d_a = nullptr;
    d_b = nullptr;
    d_c = nullptr;

    return correct
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
