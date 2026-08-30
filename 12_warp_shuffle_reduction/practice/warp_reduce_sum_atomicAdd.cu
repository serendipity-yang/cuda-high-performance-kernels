#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
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


constexpr int BLOCK_SIZE =
    256;


__inline__
__device__
float warp_reduce_sum(
    float value
)
{
    const unsigned int mask =
        0xffffffffu;


    for (
        int offset =
            warpSize / 2;

        offset > 0;

        offset /= 2
    )
    {
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
void reduce_warp_shuffle(
    const float* input,
    float* output,
    int n
)
{
    __shared__
    float warp_sums[32];


    const int tid =
        threadIdx.x;


    const int idx =
        blockIdx.x
        *
        blockDim.x
        *
        2
        +
        tid;


    float sum =
        0.0F;


    if (idx < n)
    {
        sum +=
            input[idx];
    }


    if (
        idx + blockDim.x
        <
        n
    )
    {
        sum +=
            input[
                idx
                +
                blockDim.x
            ];
    }


    sum =
        warp_reduce_sum(
            sum
        );


    const int lane =
        tid % warpSize;


    const int warp_id =
        tid / warpSize;


    if (lane == 0)
    {
        warp_sums[warp_id]
            =
            sum;
    }


    __syncthreads();


    if (warp_id == 0)
    {
        const int num_warps =
            (
                blockDim.x
                +
                warpSize
                -
                1
            )
            /
            warpSize;


        if (lane < num_warps)
        {
            sum =
                warp_sums[lane];
        }
        else
        {
            sum =
                0.0F;
        }


        sum =
            warp_reduce_sum(
                sum
            );


        if (lane == 0)
        {
            atomicAdd(
                output,
                sum
            );
        }
    }
}


int main()
{
    const int n =
        1 << 22;


    const size_t bytes =
        n * sizeof(float);


    std::vector<float>
        h_input(
            n,
            1.0F
        );


    float* d_input =
        nullptr;

    float* d_output =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemset(
            d_output,
            0,
            sizeof(float)
        )
    );


    const int blocks =
        (
            n
            +
            BLOCK_SIZE * 2
            -
            1
        )
        /
        (
            BLOCK_SIZE * 2
        );


    reduce_warp_shuffle
    <<<blocks, BLOCK_SIZE>>>(
        d_input,
        d_output,
        n
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    float gpu_sum =
        0.0F;


    CUDA_CHECK(
        cudaMemcpy(
            &gpu_sum,
            d_output,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    const float cpu_sum =
        static_cast<float>(n);


    const bool correct =
        std::fabs(
            cpu_sum
            -
            gpu_sum
        )
        <
        1e-5F;


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
        cudaFree(
            d_input
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_output
        )
    );


    return 0;
}
