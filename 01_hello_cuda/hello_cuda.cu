#include <cstdio>
#include <cuda_runtime.h>

// __global__ 表示这是一个 CUDA Kernel。
// 该函数由 CPU 发起调用，但函数体在 GPU 上执行。
__global__ void hello_cuda() {
    printf(
        "blockIdx=%d, threadIdx=%d, blockDim=%d, girdDim=%d\n",
        blockIdx.x,
        threadIdx.x,
	blockDim.x,
	gridDim.x
    );
}

int main() {
    // 启动 2 个线程块，每个线程块包含 4 个线程。
    // 总线程数量：2 × 4 = 8。
    hello_cuda<<<3, 2>>>();

    // CUDA Kernel 默认是异步执行的。
    // CPU 在这里等待 GPU 执行完成，同时检查运行错误。
    cudaError_t error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(error)
        );
        return 1;
    }

    return 0;
}
