#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


//==================================================
// CUDA Error Check
//==================================================

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
} while (0)


//==================================================
// Configuration
//==================================================

constexpr int BLOCK_SIZE =
    256;


//==================================================
// Softmax Kernel
//
// One Block -> One Row
//
// 当前学习版：
//
// cols <= BLOCK_SIZE
//
// 最典型：
// cols = 256
//==================================================

__global__
void softmax_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
)
{
    //----------------------------------------------
    // Dynamic Shared Memory
    //
    // 一个Block：
    //
    // shared[0 ... 255]
    //----------------------------------------------

    extern __shared__
    float shared[];


    //----------------------------------------------
    // 当前Thread在线程块里的编号
    //----------------------------------------------

    const int tid =
        threadIdx.x;


    //----------------------------------------------
    // 一个Block负责一行
    //----------------------------------------------

    const int row =
        blockIdx.x;


    //----------------------------------------------
    // 防止Block超过rows
    //----------------------------------------------

    if (row >= rows)
    {
        return;
    }


    //----------------------------------------------
    // 当前Thread对应这一行的第几个元素
    //----------------------------------------------

    const int col =
        tid;


    //----------------------------------------------
    // 当前元素在线性数组里的位置
    //
    // row * cols + col
    //----------------------------------------------

    const int idx =
        row
        *
        cols
        +
        col;


    //================================================
    // Step 1
    // Load Input
    //
    // 为 Max Reduction 准备
    //================================================

    float value =
        -FLT_MAX;


    //----------------------------------------------
    // 有效Thread：
    //
    // value = input[idx]
    //
    // 无效Thread：
    //
    // value = -FLT_MAX
    //
    // 这样它不会影响最大值
    //----------------------------------------------

    if (col < cols)
    {
        value =
            input[idx];
    }


    //----------------------------------------------
    // Register
    //
    // ↓
    //
    // Shared Memory
    //----------------------------------------------

    shared[tid]
        =
        value;


    //----------------------------------------------
    // 等所有Thread把数据写完
    //----------------------------------------------

    __syncthreads();


    //================================================
    // Step 2
    // Max Reduction
    //================================================

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
                =
                fmaxf(
                    shared[tid],
                    shared[
                        tid
                        +
                        stride
                    ]
                );
        }


        __syncthreads();
    }


    //----------------------------------------------
    // shared[0]
    //
    // =
    //
    // 当前整行最大值
    //----------------------------------------------

    const float row_max =
        shared[0];


    //================================================
    // Step 3
    // exp(x - max)
    //================================================

    float exp_value =
        0.0F;


    if (col < cols)
    {
        exp_value =
            expf(
                input[idx]
                -
                row_max
            );
    }


    //----------------------------------------------
    // 保存到 output
    //
    // 暂时还不是最终Softmax
    //
    // 现在只是exp值
    //----------------------------------------------

    if (col < cols)
    {
        output[idx]
            =
            exp_value;
    }


    //----------------------------------------------
    // exp值重新放Shared
    //
    // 接下来要做Sum Reduction
    //----------------------------------------------

    shared[tid]
        =
        exp_value;


    __syncthreads();


    //================================================
    // Step 4
    // Sum Reduction
    //================================================

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
                    tid
                    +
                    stride
                ];
        }


        __syncthreads();
    }


    //----------------------------------------------
    // shared[0]
    //
    // =
    //
    // sum(exp(x-max))
    //----------------------------------------------

    const float row_sum =
        shared[0];


    //================================================
    // Step 5
    // Normalize
    //================================================

    if (col < cols)
    {
        output[idx]
            =
            output[idx]
            /
            row_sum;
    }
}


//==================================================
// CPU Softmax Reference
//==================================================

void softmax_cpu(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows,
    int cols
)
{
    for (
        int row = 0;
        row < rows;
        ++row
    )
    {
        //------------------------------------------
        // Step 1
        // Find Max
        //------------------------------------------

        float row_max =
            -FLT_MAX;


        for (
            int col = 0;
            col < cols;
            ++col
        )
        {
            const int idx =
                row * cols
                +
                col;


            row_max =
                std::fmax(
                    row_max,
                    input[idx]
                );
        }


        //------------------------------------------
        // Step 2
        // exp + sum
        //------------------------------------------

        float row_sum =
            0.0F;


        for (
            int col = 0;
            col < cols;
            ++col
        )
        {
            const int idx =
                row * cols
                +
                col;


            const float value =
                std::exp(
                    input[idx]
                    -
                    row_max
                );


            output[idx]
                =
                value;


            row_sum
                +=
                value;
        }


        //------------------------------------------
        // Step 3
        // Normalize
        //------------------------------------------

        for (
            int col = 0;
            col < cols;
            ++col
        )
        {
            const int idx =
                row * cols
                +
                col;


            output[idx]
                /=
                row_sum;
        }
    }
}


//==================================================
// Main
//==================================================

int main()
{
    constexpr int rows =
        4;


    constexpr int cols =
        256;


    const int n =
        rows
        *
        cols;


    const size_t bytes =
        static_cast<size_t>(n)
        *
        sizeof(float);


    //----------------------------------------------
    // Host Data
    //----------------------------------------------

    std::vector<float>
        h_input(n);


    std::vector<float>
        h_gpu_output(
            n,
            0.0F
        );


    std::vector<float>
        h_cpu_output(
            n,
            0.0F
        );


    //----------------------------------------------
    // 构造一些简单输入
    //----------------------------------------------

    for (
        int row = 0;
        row < rows;
        ++row
    )
    {
        for (
            int col = 0;
            col < cols;
            ++col
        )
        {
            const int idx =
                row * cols
                +
                col;


            h_input[idx]
                =
                static_cast<float>(
                    col % 10
                )
                *
                0.1F;
        }
    }


    //----------------------------------------------
    // CPU Reference
    //----------------------------------------------

    softmax_cpu(
        h_input,
        h_cpu_output,
        rows,
        cols
    );


    //----------------------------------------------
    // Device Memory
    //----------------------------------------------

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
            bytes
        )
    );


    //----------------------------------------------
    // CPU -> GPU
    //----------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    //----------------------------------------------
    // One Block -> One Row
    //
    // rows = 4
    //
    // 所以：
    //
    // 4 Blocks
    //----------------------------------------------

    const int blocks =
        rows;


    //----------------------------------------------
    // 每个Block Shared Memory：
    //
    // 256 floats
    //
    // =
    //
    // 1024 bytes
    //----------------------------------------------

    const size_t shared_bytes =
        BLOCK_SIZE
        *
        sizeof(float);


    //----------------------------------------------
    // Launch
    //----------------------------------------------

    softmax_kernel
    <<<
        blocks,
        BLOCK_SIZE,
        shared_bytes
    >>>(
        d_input,
        d_output,
        rows,
        cols
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    //----------------------------------------------
    // GPU -> CPU
    //----------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_gpu_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //----------------------------------------------
    // Correctness
    //----------------------------------------------

    bool correct =
        true;


    for (
        int i = 0;
        i < n;
        ++i
    )
    {
        const float diff =
            std::fabs(
                h_gpu_output[i]
                -
                h_cpu_output[i]
            );


        if (diff > 1e-5F)
        {
            correct =
                false;


            std::cerr
                << "Mismatch at i = "
                << i
                << '\n';


            std::cerr
                << "CPU = "
                << h_cpu_output[i]
                << '\n';


            std::cerr
                << "GPU = "
                << h_gpu_output[i]
                << '\n';


            break;
        }
    }


    //----------------------------------------------
    // 每行Softmax理论上应该加起来≈1
    //----------------------------------------------

    for (
        int row = 0;
        row < rows;
        ++row
    )
    {
        float sum =
            0.0F;


        for (
            int col = 0;
            col < cols;
            ++col
        )
        {
            sum
                +=
                h_gpu_output[
                    row * cols
                    +
                    col
                ];
        }


        std::cout
            << "row "
            << row
            << " sum = "
            << sum
            << '\n';
    }


    std::cout
        << "correct = "
        << std::boolalpha
        << correct
        << '\n';


    //----------------------------------------------
    // Free
    //----------------------------------------------

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
