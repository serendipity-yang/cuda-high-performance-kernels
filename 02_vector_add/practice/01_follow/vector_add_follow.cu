#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t error = (call);                             \
        if (error != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                         \
                      << cudaGetErrorString(error)              \
                      << " at line " << __LINE__ << std::endl;  \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)


__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n) {

    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    printf(
        "block=%d, thread=%d, idx=%d\n",
        blockIdx.x,
        threadIdx.x,
        idx
    );

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 10;
    const std::size_t bytes =
        n * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 10);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

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

    const int threads_per_block = 4;
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

    vector_add<<<blocks, threads_per_block>>>(
        d_a,
        d_b,
        d_c,
        n
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < n; ++i) {
        std::cout
            << h_a[i]
            << " + "
            << h_b[i]
            << " = "
            << h_c[i]
            << '\n';
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
