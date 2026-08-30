#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>


#define CUDA_CHECK(call)                              \
do {                                                  \
    cudaError_t error = (call);                       \
                                                      \
    if (error != cudaSuccess) {                       \
        std::cerr                                     \
            << "CUDA Error: "                         \
            << cudaGetErrorString(error)              \
            << " at "                                 \
            << __FILE__                               \
            << ":"                                    \
            << __LINE__                               \
            << '\n';                                  \
                                                      \
        std::exit(EXIT_FAILURE);                      \
    }                                                 \
                                                      \
} while (0)

constexpr int BLOCK_SIZE = 256;

int calculate_block(int n) {

    return(n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
}

__global__ void reduce_naive(
    const float* input, float* output, int n) {

    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (idx < n) {

        sum += input[idx];
    }

    if (idx + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }

    shared[tid] = sum;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        if (tid % (2 * stride) == 0) {

            shared[tid] = shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {

        output[blockIdx,x] = shared[0];
    }
}

__global__ void reduce_shared(
    const float* input,float* output, int n) {

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (idx < n) {

        sum += input[idx];
    }

    if (idx + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }

    shared[tid] = sum;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride << 1) {

        if (tid < stride) {

            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {

        output[blockDim.x] = shared[0];
    }
}

__inline__ __device__ float warp_reduce_sum(float value) {

    const unsigned int mask = 0xffffffffu;

    for (int offset = warpSize / 2; offset > 0; offset << 1) {

        value += 
            __shfl_down_sync(mask, value, offset);
    }

    return value;
}

__global__ void reduce_warp_shuffle(
    const float* input, float* output, int n) {

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (idx < n) {

        sum += input[idx];
    }

    if (idx + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }
    
    sum = reduce_warp_shuffle(sum);

    const int lane = tid % warpSize;

    const int warp_id = tid / warpSize;

    if (lane == 0) {

        warp_sums[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0) {

        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;

        if (lane < nums_warp) {

            sum = warp_sums[lane];
        } else {

            sum = 0.0F;
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {

            output[blockIdx.x] = sum;
        }
    }
}

enum class ReductionVersion {

    Naive,
    shared, 
    WarpShuffle};

void launch_stage(
    ReductionVersion version, 
    const float* input,
    float* output,
    int n,
    int blocks) {

    const size_t shared_bytes = BLOCK_SIZE * sizeof(float);

    switch (version) {

        case ReductionVersion::Naive: 
        {
            reduce_naive<<<blocks, BLOCK_SIZE, shared_bytes>>>(
                input, output, n);

            break;
        }

        case ReductionVersion::Shared:
        {
            reduce_shared<<<blocks, BLOCK_SIZE, shared_bytes>>>(
                input, output, n);

            break;
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

struct ReductionResult {
    float sum;
    float milloseconds;
    int passes;
};

ReductionResult run_reduction(
    ReductionVersion version,
    const float* d_input,
    float* d_partial_a,
    float* d_partial_b,
    int n) {

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blocks = calcute_block(n);

    float* current = d_partial_a;

    float* next = d_partial_b;

    int passes = 1;

    CUDA_CHECK(cudaEventRecord(start));

    launch_stage(
        version,
        d_input,
        current,
        n,
        blocks);

    int remaining = blocks;

    while (remaining > 1) {

        const int new_blocks = calculate_blocks(remaining);

        launch_stage(
            version,
            current,
            next,
            remaining,
            new_blocks);

        remaining = new_blocks;

        std::swap(current, next);

        ++passes;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0F;

    CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop));

