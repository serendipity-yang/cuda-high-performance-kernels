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


constexpr int BLOCK_SIZE =
    256;


//==================================================
// Utility
//==================================================

int calculate_blocks(
    int n
)
{
    return
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
}


//==================================================
// 10 Naive Reduction
//==================================================

__global__
void reduce_naive(
    const float* input,
    float* output,
    int n
)
{
    extern __shared__
    float shared[];


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
                idx + blockDim.x
            ];
    }


    shared[tid] =
        sum;


    __syncthreads();


    for (
        int stride = 1;

        stride < blockDim.x;

        stride *= 2
    )
    {
        if (
            tid
            %
            (2 * stride)
            ==
            0
        )
        {
            shared[tid]
                +=
                shared[
                    tid + stride
                ];
        }


        __syncthreads();
    }


    if (tid == 0)
    {
        output[blockIdx.x]
            =
            shared[0];
    }
}


//==================================================
// 11 Shared Memory Reduction
//==================================================

__global__
void reduce_shared(
    const float* input,
    float* output,
    int n
)
{
    extern __shared__
    float shared[];


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
                idx + blockDim.x
            ];
    }


    shared[tid] =
        sum;


    __syncthreads();


    for (
        int stride =
            blockDim.x / 2;

        stride > 0;

        stride /= 2
    )
    {
        if (tid < stride)
        {
            shared[tid]
                +=
                shared[
                    tid + stride
                ];
        }


        __syncthreads();
    }


    if (tid == 0)
    {
        output[blockIdx.x]
            =
            shared[0];
    }
}


//==================================================
// Warp helper
//==================================================

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


//==================================================
// 12 Warp Shuffle Reduction
//==================================================

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
                idx + blockDim.x
            ];
    }


    //--------------------------------
    // 每Warp先归约
    //--------------------------------

    sum =
        warp_reduce_sum(
            sum
        );


    const int lane =
        tid % warpSize;


    const int warp_id =
        tid / warpSize;


    //--------------------------------
    // 每Warp的lane0写结果
    //--------------------------------

    if (lane == 0)
    {
        warp_sums[warp_id]
            =
            sum;
    }


    __syncthreads();


    //--------------------------------
    // Warp0完成第二级归约
    //--------------------------------

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
            output[blockIdx.x]
                =
                sum;
        }
    }
}


//==================================================
// Version
//==================================================

enum class ReductionVersion
{
    Naive,
    Shared,
    WarpShuffle
};


//==================================================
// Launch one reduction stage
//==================================================

void launch_stage(
    ReductionVersion version,
    const float* input,
    float* output,
    int n,
    int blocks
)
{
    const size_t shared_bytes =
        BLOCK_SIZE
        *
        sizeof(float);


    switch (version)
    {
        case ReductionVersion::Naive:
        {
            reduce_naive
            <<<
                blocks,
                BLOCK_SIZE,
                shared_bytes
            >>>(
                input,
                output,
                n
            );

            break;
        }


        case ReductionVersion::Shared:
        {
            reduce_shared
            <<<
                blocks,
                BLOCK_SIZE,
                shared_bytes
            >>>(
                input,
                output,
                n
            );

            break;
        }


        case ReductionVersion::WarpShuffle:
        {
            reduce_warp_shuffle
            <<<
                blocks,
                BLOCK_SIZE
            >>>(
                input,
                output,
                n
            );

            break;
        }
    }


    CUDA_CHECK(
        cudaGetLastError()
    );
}


//==================================================
// Result
//==================================================

struct ReductionResult
{
    float sum;
    float milliseconds;
    int passes;
};


//==================================================
// Run complete multi-pass reduction
//==================================================

ReductionResult run_reduction(
    ReductionVersion version,
    const float* d_input,
    float* d_partial_a,
    float* d_partial_b,
    int n
)
{
    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(
            &start
        )
    );


    CUDA_CHECK(
        cudaEventCreate(
            &stop
        )
    );


    //------------------------------------------
    // First pass
    //------------------------------------------

    int blocks =
        calculate_blocks(
            n
        );


    float* current =
        d_partial_a;


    float* next =
        d_partial_b;


    int passes =
        1;


    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );


    launch_stage(
        version,
        d_input,
        current,
        n,
        blocks
    );


    int remaining =
        blocks;


    //------------------------------------------
    // Remaining passes
    //------------------------------------------

    while (remaining > 1)
    {
        const int new_blocks =
            calculate_blocks(
                remaining
            );


        launch_stage(
            version,
            current,
            next,
            remaining,
            new_blocks
        );


        remaining =
            new_blocks;


        std::swap(
            current,
            next
        );


        ++passes;
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop
        )
    );


    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
    );


    float milliseconds =
        0.0F;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );


    //------------------------------------------
    // Final result is current[0]
    //------------------------------------------

    float result =
        0.0F;


    CUDA_CHECK(
        cudaMemcpy(
            &result,
            current,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaEventDestroy(
            start
        )
    );


    CUDA_CHECK(
        cudaEventDestroy(
            stop
        )
    );


    return {
        result,
        milliseconds,
        passes
    };
}


//==================================================
// Main
//==================================================

int main()
{
    const int n =
        1 << 22;


    const size_t bytes =
        static_cast<size_t>(n)
        *
        sizeof(float);


    //------------------------------------------
    // Host input
    //------------------------------------------

    std::vector<float>
        h_input(
            n,
            1.0F
        );


    //------------------------------------------
    // Device input
    //------------------------------------------

    float* d_input =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            bytes
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


    //------------------------------------------
    // Largest partial buffer needed
    //------------------------------------------

    const int first_blocks =
        calculate_blocks(
            n
        );


    float* d_partial_a =
        nullptr;

    float* d_partial_b =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_partial_a,
            first_blocks
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_partial_b,
            first_blocks
            *
            sizeof(float)
        )
    );


    //------------------------------------------
    // CPU reference
    //------------------------------------------

    const float expected =
        static_cast<float>(
            n
        );


    //------------------------------------------
    // Run three versions
    //------------------------------------------

    const ReductionResult naive =
        run_reduction(
            ReductionVersion::Naive,
            d_input,
            d_partial_a,
            d_partial_b,
            n
        );


    const ReductionResult shared =
        run_reduction(
            ReductionVersion::Shared,
            d_input,
            d_partial_a,
            d_partial_b,
            n
        );


    const ReductionResult shuffle =
        run_reduction(
            ReductionVersion::WarpShuffle,
            d_input,
            d_partial_a,
            d_partial_b,
            n
        );


    //------------------------------------------
    // Correctness
    //------------------------------------------

    const bool naive_correct =
        std::fabs(
            naive.sum
            -
            expected
        )
        <
        1e-5F;


    const bool shared_correct =
        std::fabs(
            shared.sum
            -
            expected
        )
        <
        1e-5F;


    const bool shuffle_correct =
        std::fabs(
            shuffle.sum
            -
            expected
        )
        <
        1e-5F;


    //------------------------------------------
    // Print
    //------------------------------------------

    std::cout
        << "n = "
        << n
        << '\n';


    std::cout
        << "BLOCK_SIZE = "
        << BLOCK_SIZE
        << '\n';


    std::cout
        << "elements/block = "
        << BLOCK_SIZE * 2
        << '\n';


    std::cout
        << "first blocks = "
        << first_blocks
        << "\n\n";


    std::cout
        << std::boolalpha;


    std::cout
        << "Naive:\n"
        << "  sum = "
        << naive.sum
        << '\n'
        << "  correct = "
        << naive_correct
        << '\n'
        << "  passes = "
        << naive.passes
        << '\n'
        << "  time = "
        << naive.milliseconds
        << " ms\n\n";


    std::cout
        << "Shared:\n"
        << "  sum = "
        << shared.sum
        << '\n'
        << "  correct = "
        << shared_correct
        << '\n'
        << "  passes = "
        << shared.passes
        << '\n'
        << "  time = "
        << shared.milliseconds
        << " ms\n\n";


    std::cout
        << "Warp Shuffle:\n"
        << "  sum = "
        << shuffle.sum
        << '\n'
        << "  correct = "
        << shuffle_correct
        << '\n'
        << "  passes = "
        << shuffle.passes
        << '\n'
        << "  time = "
        << shuffle.milliseconds
        << " ms\n";


    //------------------------------------------
    // Free
    //------------------------------------------

    CUDA_CHECK(
        cudaFree(
            d_input
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_partial_a
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_partial_b
        )
    );


    return 0;
}
