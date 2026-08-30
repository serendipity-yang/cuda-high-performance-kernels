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
// 09-A
// Bank Conflict Version
//==================================================

__global__
void transpose_conflict(
    const float* input,
    float* output,
    int width,
    int height
)
{
    //----------------------------------------------
    // stride = 32
    //
    // tile[tx][ty]读取时
    // 会发生Bank Conflict
    //----------------------------------------------

    __shared__
    float tile[
        TILE_DIM
    ][
        TILE_DIM
    ];


    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


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
    // Global -> Shared
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


    __syncthreads();


    //----------------------------------------------
    // Output coordinates
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
    // Problem:
    //
    // tile[tx][ty]
    //
    // stride = 32 floats
    //
    // 同一个Warp可能全部访问同一Bank
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
// 09-B
// Padding Version
//==================================================

__global__
void transpose_padding(
    const float* input,
    float* output,
    int width,
    int height
)
{
    //----------------------------------------------
    // 关键变化：
    //
    // 32
    // ↓
    // 33
    //
    // 逻辑Tile仍然是32×32
    //
    // 物理row stride变成33
    //----------------------------------------------

    __shared__
    float tile[
        TILE_DIM
    ][
        TILE_DIM + 1
    ];


    const int tx =
        threadIdx.x;

    const int ty =
        threadIdx.y;


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
    // Global -> Shared
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


    __syncthreads();


    //----------------------------------------------
    // Output coordinates
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
    // 仍然逻辑上：
    //
    // tile[tx][ty]
    //
    // 但是物理stride已经变成33
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
        2048;

    constexpr int height =
        2048;


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
        h_conflict(
            elements,
            0.0F
        );


    std::vector<float>
        h_padding(
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
    // CUDA Events
    //----------------------------------------------

    cudaEvent_t start =
        nullptr;

    cudaEvent_t stop =
        nullptr;


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


    //----------------------------------------------
    // Warm-up
    //----------------------------------------------

    transpose_padding
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
        cudaDeviceSynchronize()
    );


    //----------------------------------------------
    // Test 1
    // Bank Conflict
    //----------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );


    transpose_conflict
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
        cudaEventRecord(
            stop
        )
    );


    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
    );


    float conflict_ms =
        0.0F;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &conflict_ms,
            start,
            stop
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_conflict.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //----------------------------------------------
    // Test 2
    // Padding
    //----------------------------------------------

    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );


    transpose_padding
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
        cudaEventRecord(
            stop
        )
    );


    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
    );


    float padding_ms =
        0.0F;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &padding_ms,
            start,
            stop
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_padding.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //----------------------------------------------
    // Correctness
    //----------------------------------------------

    bool conflict_correct =
        true;

    bool padding_correct =
        true;


    for (
        int row = 0;
        row < height;
        ++row
    )
    {
        for (
            int col = 0;
            col < width;
            ++col
        )
        {
            const float expected =
                h_input[
                    row * width
                    +
                    col
                ];


            const int output_index =
                col
                *
                height
                +
                row;


            if (
                std::fabs(
                    expected
                    -
                    h_conflict[
                        output_index
                    ]
                )
                >
                1e-6F
            )
            {
                conflict_correct =
                    false;
            }


            if (
                std::fabs(
                    expected
                    -
                    h_padding[
                        output_index
                    ]
                )
                >
                1e-6F
            )
            {
                padding_correct =
                    false;
            }


            if (
                !conflict_correct
                ||
                !padding_correct
            )
            {
                break;
            }
        }


        if (
            !conflict_correct
            ||
            !padding_correct
        )
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
        << "Conflict correct = "
        << conflict_correct
        << '\n';


    std::cout
        << "Padding correct = "
        << padding_correct
        << '\n';


    std::cout
        << "Conflict time = "
        << conflict_ms
        << " ms\n";


    std::cout
        << "Padding time = "
        << padding_ms
        << " ms\n";


    //----------------------------------------------
    // Destroy Event
    //----------------------------------------------

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
