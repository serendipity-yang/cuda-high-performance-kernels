#include <cuda_runtime.h>

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

constexpr int TILE_DIM =
    32;


//==================================================
// 07 Naive Matrix Transpose
//==================================================

__global__
void transpose_naive(
    const float* input,
    float* output,
    int width,
    int height
)
{
    //----------------------------------------------
    // 当前 Thread 对应输入矩阵中的：
    //
    // x = column
    // y = row
    //----------------------------------------------

    const int x =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    const int y =
        blockIdx.y
        *
        blockDim.y
        +
        threadIdx.y;


    //----------------------------------------------
    // 边界检查
    //----------------------------------------------

    if (
        x < width
        &&
        y < height
    )
    {
        //------------------------------------------
        // input[y][x]
        //
        // ↓ transpose
        //
        // output[x][y]
        //------------------------------------------

        output[
            x * height
            +
            y
        ]
        =
        input[
            y * width
            +
            x
        ];
    }
}


//==================================================
// Main
//==================================================

int main()
{
    //----------------------------------------------
    // 输入矩阵尺寸：
    //
    // height rows
    // width cols
    //----------------------------------------------

    constexpr int width =
        2050;

    constexpr int height =
        1030;


    const size_t elements =
        static_cast<size_t>(
            width
        )
        *
        height;


    const size_t bytes =
        elements
        *
        sizeof(float);


    //----------------------------------------------
    // Host Input
    //----------------------------------------------

    std::vector<float>
        h_input(
            elements
        );


    //----------------------------------------------
    // 给每个位置一个容易检查的值
    //----------------------------------------------

    for (
        size_t i = 0;
        i < elements;
        ++i
    )
    {
        h_input[i] =
            static_cast<float>(i);
    }


    //----------------------------------------------
    // Host Output
    //----------------------------------------------

    std::vector<float>
        h_output(
            elements,
            0.0F
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
    // Block
    //
    // 32 × 32
    // =
    // 1024 Threads
    //----------------------------------------------

    dim3 block(
        TILE_DIM,
        TILE_DIM
    );


    //----------------------------------------------
    // Grid
    //
    // x方向覆盖 width
    // y方向覆盖 height
    //----------------------------------------------

    dim3 grid(
        (
            width
            +
            TILE_DIM
            -
            1
        )
        /
        TILE_DIM,

        (
            height
            +
            TILE_DIM
            -
            1
        )
        /
        TILE_DIM
    );


    //----------------------------------------------
    // Launch
    //----------------------------------------------

    transpose_naive
    <<<
        grid,
        block
    >>>(
        d_input,
        d_output,
        width,
        height
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
            h_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //----------------------------------------------
    // Correctness Check
    //----------------------------------------------

    bool correct =
        true;


    for (
        int input_row = 0;
        input_row < height;
        ++input_row
    )
    {
        for (
            int input_col = 0;
            input_col < width;
            ++input_col
        )
        {
            //--------------------------------------
            // 原矩阵：
            //
            // input[row][col]
            //--------------------------------------

            const float expected =
                h_input[
                    input_row
                    *
                    width
                    +
                    input_col
                ];


            //--------------------------------------
            // 转置后：
            //
            // output[col][row]
            //
            // 输出每行长度 = height
            //--------------------------------------

            const float actual =
                h_output[
                    input_col
                    *
                    height
                    +
                    input_row
                ];


            if (
                std::fabs(
                    expected
                    -
                    actual
                )
                >
                1e-6F
            )
            {
                correct =
                    false;


                std::cerr
                    << "Mismatch:"
                    << " row = "
                    << input_row
                    << ", col = "
                    << input_col
                    << '\n';


                break;
            }
        }


        if (!correct)
        {
            break;
        }
    }


    //----------------------------------------------
    // Print
    //----------------------------------------------

    std::cout
        << std::boolalpha;


    std::cout
        << "input size = "
        << height
        << " x "
        << width
        << '\n';


    std::cout
        << "output size = "
        << width
        << " x "
        << height
        << '\n';


    std::cout
        << "correct = "
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
