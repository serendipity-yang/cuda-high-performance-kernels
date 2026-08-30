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


constexpr int TILE_DIM = 32;


//==================================================
// Version 1
// Shared Memory Transpose
//
// tile[32][32]
//
// 转置读取时：
// tile[tx][ty]
//
// 会产生严重 Bank Conflict
//==================================================

__global__
void transpose_bank_conflict(
    const float* input,
    float* output,
    int width,
    int height
)
{
    __shared__
    float tile[TILE_DIM][TILE_DIM];


    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    //------------------------------------------------
    // 输入矩阵中的全局坐标
    //------------------------------------------------

    const int x =
        blockIdx.x * TILE_DIM
        +
        tx;

    const int y =
        blockIdx.y * TILE_DIM
        +
        ty;


    //------------------------------------------------
    // Global Memory -> Shared Memory
    //
    // warp横着读取Global Memory
    //
    // tile[ty][tx]
    //
    // warp也横着写Shared Memory
    //------------------------------------------------

    if (x < width &&
        y < height)
    {
        tile[ty][tx]
            =
            input[
                y * width + x
            ];
    }


    //------------------------------------------------
    // 必须保证整个Tile已经加载完成
    //------------------------------------------------

    __syncthreads();


    //------------------------------------------------
    // 输出Tile坐标交换
    //------------------------------------------------

    const int out_x =
        blockIdx.y * TILE_DIM
        +
        tx;

    const int out_y =
        blockIdx.x * TILE_DIM
        +
        ty;


    //------------------------------------------------
    // Shared Memory -> Global Memory
    //
    // 关键：
    //
    // tile[tx][ty]
    //
    // warp会竖着读取Shared Memory
    //
    // tile[32][32]时会Bank Conflict
    //------------------------------------------------

    if (out_x < height &&
        out_y < width)
    {
        output[
            out_y * height
            +
            out_x
        ]
        =
        tile[tx][ty];
    }
}


//==================================================
// Version 2
// Shared Memory + Padding
//
// tile[32][33]
//
// 只多一列Padding
//
// 让竖向读取分散到不同Bank
//==================================================

__global__
void transpose_padding(
    const float* input,
    float* output,
    int width,
    int height
)
{
    //------------------------------------------------
    // 这里是整个09最重要的一行
    //------------------------------------------------

    __shared__
    float tile[TILE_DIM][TILE_DIM + 1];


    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    //------------------------------------------------
    // 输入Global坐标
    //------------------------------------------------

    const int x =
        blockIdx.x * TILE_DIM
        +
        tx;

    const int y =
        blockIdx.y * TILE_DIM
        +
        ty;


    //------------------------------------------------
    // Global -> Shared
    //------------------------------------------------

    if (x < width &&
        y < height)
    {
        tile[ty][tx]
            =
            input[
                y * width + x
            ];
    }


    //------------------------------------------------
    // 等整个Block
    //------------------------------------------------

    __syncthreads();


    //------------------------------------------------
    // 输出Tile位置交换
    //------------------------------------------------

    const int out_x =
        blockIdx.y * TILE_DIM
        +
        tx;

    const int out_y =
        blockIdx.x * TILE_DIM
        +
        ty;


    //------------------------------------------------
    // Shared -> Global
    //
    // 仍然是 tile[tx][ty]
    //
    // 但是底层stride从32变成33
    //------------------------------------------------

    if (out_x < height &&
        out_y < width)
    {
        output[
            out_y * height
            +
            out_x
        ]
        =
        tile[tx][ty];
    }
}


//==================================================
// CPU Correctness Check
//==================================================

bool check_transpose(
    const std::vector<float>& input,
    const std::vector<float>& output,
    int width,
    int height
)
{
    for (int input_row = 0;
         input_row < height;
         ++input_row)
    {
        for (int input_col = 0;
             input_col < width;
             ++input_col)
        {
            //------------------------------------------------
            // 输入：
            //
            // input[row][col]
            //------------------------------------------------

            const float expected =
                input[
                    input_row * width
                    +
                    input_col
                ];


            //------------------------------------------------
            // 转置以后：
            //
            // output[col][row]
            //------------------------------------------------

            const float actual =
                output[
                    input_col * height
                    +
                    input_row
                ];


            if (
                std::fabs(
                    expected - actual
                )
                >
                1e-6F
            )
            {
                std::cerr
                    << "Mismatch at input("
                    << input_row
                    << ", "
                    << input_col
                    << ")\n";

                std::cerr
                    << "expected = "
                    << expected
                    << '\n';

                std::cerr
                    << "actual = "
                    << actual
                    << '\n';


                return false;
            }
        }
    }


    return true;
}


//==================================================
// Main
//==================================================

int main()
{
    constexpr int width =
        1024;

    constexpr int height =
        1024;


    const int num_elements =
        width * height;


    const size_t bytes =
        num_elements
        *
        sizeof(float);


    //------------------------------------------------
    // Host Input
    //------------------------------------------------

    std::vector<float>
        h_input(
            num_elements
        );


    for (int i = 0;
         i < num_elements;
         ++i)
    {
        h_input[i]
            =
            static_cast<float>(i);
    }


    //------------------------------------------------
    // Host Output
    //------------------------------------------------

    std::vector<float>
        h_output(
            num_elements,
            0.0F
        );


    //------------------------------------------------
    // Device Memory
    //------------------------------------------------

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


    //------------------------------------------------
    // CPU -> GPU
    //------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    //------------------------------------------------
    // Block
    //
    // 32 x 32
    //
    // = 1024 Threads
    //------------------------------------------------

    dim3 block(
        TILE_DIM,
        TILE_DIM
    );


    //------------------------------------------------
    // Grid
    //
    // 每一个Block负责一个32x32 Tile
    //------------------------------------------------

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


    //------------------------------------------------
    // Version 1
    // Bank Conflict
    //------------------------------------------------

    transpose_bank_conflict
    <<<grid, block>>>(
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


    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    const bool conflict_correct =
        check_transpose(
            h_input,
            h_output,
            width,
            height
        );


    std::cout
        << "Bank conflict version correct = "
        << std::boolalpha
        << conflict_correct
        << '\n';


    //------------------------------------------------
    // Version 2
    // Padding
    //------------------------------------------------

    transpose_padding
    <<<grid, block>>>(
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


    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    const bool padding_correct =
        check_transpose(
            h_input,
            h_output,
            width,
            height
        );


    std::cout
        << "Padding version correct = "
        << std::boolalpha
        << padding_correct
        << '\n';


    //------------------------------------------------
    // Free GPU Memory
    //------------------------------------------------

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
