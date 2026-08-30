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
} while (0)


constexpr int TILE_DIM =
    32;


//==================================================
// 08 Shared Memory Tiled Transpose
//==================================================

__global__
void transpose_tiled(
    const float* input,
    float* output,
    int width,
    int height
)
{
    //----------------------------------------------
    // 每一个Block拥有自己的一块：
    //
    // 32 × 32 Shared Memory
    //----------------------------------------------

    __shared__
    float tile[
        TILE_DIM
    ][
        TILE_DIM
    ];


    //----------------------------------------------
    // Thread在Tile内部的位置
    //----------------------------------------------

    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


    //----------------------------------------------
    // 输入矩阵坐标
    //----------------------------------------------

    const int x =
        blockIdx.x
        *
        TILE_DIM
        +
        tx;


    const int y =
        blockIdx.y
        *
        TILE_DIM
        +
        ty;


    //----------------------------------------------
    // Step 1
    //
    // Global Memory
    //
    // ↓
    //
    // Shared Memory
    //
    // tile[ty][tx]
    //----------------------------------------------

    if (
        x < width
        &&
        y < height
    )
    {
        tile[ty][tx]
            =
            input[
                y * width
                +
                x
            ];
    }


    //----------------------------------------------
    // 必须同步
    //
    // 因为之后：
    //
    // 一个Thread会读取
    // 其他Thread写入的tile元素
    //----------------------------------------------

    __syncthreads();


    //----------------------------------------------
    // Step 2
    //
    // 转置以后 Block 坐标交换
    //----------------------------------------------

    const int out_x =
        blockIdx.y
        *
        TILE_DIM
        +
        tx;


    const int out_y =
        blockIdx.x
        *
        TILE_DIM
        +
        ty;


    //----------------------------------------------
    // Shared Memory
    //
    // tile[tx][ty]
    //
    // 注意这里把索引交换了
    //----------------------------------------------

    if (
        out_x < height
        &&
        out_y < width
    )
    {
        output[
            out_y
            *
            height
            +
            out_x
        ]
        =
        tile[tx][ty];
    }
}


//==================================================
// Main
//==================================================

int main()
{
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
    // Host
    //----------------------------------------------

    std::vector<float>
        h_input(
            elements
        );


    std::vector<float>
        h_output(
            elements,
            0.0F
        );


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
    // Device
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


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    //----------------------------------------------
    // Block / Grid
    //----------------------------------------------

    dim3 block(
        TILE_DIM,
        TILE_DIM
    );


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

    transpose_tiled
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
    // Correctness
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
            const float expected =
                h_input[
                    input_row
                    *
                    width
                    +
                    input_col
                ];


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
