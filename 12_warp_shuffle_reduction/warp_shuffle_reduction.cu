#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                              \
    do {                                              \
        cudaError_t error = (call);                   \
        if (error != cudaSuccess) {                   \
            std::cerr                                 \
                << "CUDA error: "                     \
                << cudaGetErrorString(error)          \
                << " at "                             \
                << __FILE__                           \
                << ":"                                \
                << __LINE__                           \
                << '\n';                              \
            std::exit(EXIT_FAILURE);                  \
        }                                             \
    } while (0)


__inline__ __device__
float warp_reduce_sum(float value) {

    const unsigned int mask =
        0xffffffffu;

    for (int offset = warpSize / 2;
         offset > 0;
         offset /= 2) {

        value +=
            __shfl_down_sync(
                mask,
                value,
                offset
            );
    }

    return value;
}


__global__
void warp_shuffle_reduction(
    const float* input,
    float* block_sums,
    int n
) {

    __shared__
    float warp_sums[32];

    const int tid =
        threadIdx.x;

    const int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    const int lane =
        tid % warpSize;

    const int warp_id =
        tid / warpSize;


    float value = 0.0F;

    if (idx < n) {
        value = input[idx];
    }


    value =
        warp_reduce_sum(value);


    if (lane == 0) {
        warp_sums[warp_id] =
            value;
    }


    __syncthreads();


    const int num_warps =
        blockDim.x / warpSize;


    if (warp_id == 0) {

        if (lane < num_warps) {

            value =
                warp_sums[lane];

        } else {

            value =
                0.0F;
        }


        value =
            warp_reduce_sum(value);


        if (lane == 0) {

            block_sums[blockIdx.x] =
                value;
        }
    }
}


int main() {

    constexpr int n =
        1 << 22;

    constexpr int threads_per_block =
        256;


    const int blocks =
        (
            n
            + threads_per_block
            - 1
        )
        / threads_per_block;


    std::cout
        << "n = "
        << n
        << '\n';

    std::cout
        << "threads per block = "
        << threads_per_block
        << '\n';

    std::cout
        << "blocks = "
        << blocks
        << '\n';


    std::vector<float>
        h_input(
            n,
            1.0F
        );

    std::vector<float>
        h_block_sums(
            blocks,
            0.0F
        );


    float* d_input =
        nullptr;

    float* d_block_sums =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_input
            ),
            n * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_block_sums
            ),
            blocks * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    warp_shuffle_reduction
        <<<blocks, threads_per_block>>>(
            d_input,
            d_block_sums,
            n
        );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_block_sums.data(),
            d_block_sums,
            blocks * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    float gpu_sum =
        0.0F;

    for (float value : h_block_sums) {
        gpu_sum += value;
    }


    float cpu_sum =
        0.0F;

    for (float value : h_input) {
        cpu_sum += value;
    }


    const bool correct =
        std::fabs(
            cpu_sum - gpu_sum
        ) < 1e-5F;


    std::cout
        << "CPU sum = "
        << cpu_sum
        << '\n';

    std::cout
        << "GPU sum = "
        << gpu_sum
        << '\n';

    std::cout
        << "correct = "
        << std::boolalpha
        << correct
        << '\n';


    CUDA_CHECK(
        cudaFree(d_input)
    );

    CUDA_CHECK(
        cudaFree(d_block_sums)
    );


    return 0;
}
