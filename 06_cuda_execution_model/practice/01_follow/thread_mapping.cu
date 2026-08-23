#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                \
    do {                                                \
        cudaError_t error = (call);                     \
        if (error != cudaSuccess) {                     \
            std::cerr << "CUDA error: "                 \
                      << cudaGetErrorString(error)      \
                      << " at " << __FILE__             \
                      << ':' << __LINE__                \
                      << '\n';                          \
            std::exit(EXIT_FAILURE);                    \
        }                                               \
    } while (0)

__global__ void print_thread_mapping() {
    const int tid = 
        threadIdx.x;

    const int global_idx = 
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int warp_id = 
        tid / warpSize;

    const int lane_id = 
        tid % warpSize;

    printf(
        "block =%u thread =%d global=%d warp=%d lane=%d\n",
        blockIdx.x,
        tid,
        global_idx,
        warp_id,
        lane_id
    );
}

int main() {

    const int blocks = 
        2;

    const int threads_per_block =
        64;

    std::cout 
        << "blocks = "
        << blocks
        << '\n'
        << "threads per block = "
        << threads_per_block
        << '\n'
        << "warp size = "
        << 32
        << '\n';

    print_thread_mapping<<<
        blocks,
        threads_per_block
    >>>();

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
