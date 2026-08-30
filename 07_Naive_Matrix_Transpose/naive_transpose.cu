#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                              \
do {                                                  \
    cudaError_t error = (call);                       \
                                                      \
    if (error != cudaSuccess) {                       \
                                                      \
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


//==================================================
// Naive Matrix Transpose Kernel
//==================================================

__global__
void transpose_naive(
    const float* input,
    float* output,
    int width,
    int height
)
{
    const int x =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y
        +
        threadIdx.y;


    if (x < width && y < height) {

        const int input_index =
            y * width + x;

        const int output_index =
            x * height + y;


        output[output_index] =
            input[input_index];
    }
}


//==================================================
// Main
//==================================================

int main()
{
    constexpr int width =
        4;

    constexpr int height =
        4;


    const int num_elements =
        width * height;

    const size_t bytes =
        num_elements * sizeof(float);


    //------------------------------------------------
    // Host Input
    //------------------------------------------------

    std::vector<float>
        h_input(
            num_elements
        );

    for (int i = 0;
         i < num_elements;
         ++i) {

        h_input[i] =
            static_cast<float>(i + 1);
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
    // Block / Grid
    //------------------------------------------------

    constexpr int BLOCK_SIZE =
        2;


    dim3 block(
        BLOCK_SIZE,
        BLOCK_SIZE
    );


    dim3 grid(
        (width + block.x - 1)
        /
        block.x,

        (height + block.y - 1)
        /
        block.y
    );


    //------------------------------------------------
    // Launch Kernel
    //------------------------------------------------

    transpose_naive
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


    //------------------------------------------------
    // GPU -> CPU
    //------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //------------------------------------------------
    // Print Input
    //------------------------------------------------

    std::cout
        << "Input:\n";

    for (int y = 0;
         y < height;
         ++y) {

        for (int x = 0;
             x < width;
             ++x) {

            std::cout
                << h_input[
                    y * width + x
                ]
                << '\t';
        }

        std::cout
            << '\n';
    }


    //------------------------------------------------
    // Print Output
    //------------------------------------------------

    std::cout
        << "\nTranspose:\n";

    for (int y = 0;
         y < width;
         ++y) {

        for (int x = 0;
             x < height;
             ++x) {

            std::cout
                << h_output[
                    y * height + x
                ]
                << '\t';
        }

        std::cout
            << '\n';
    }


    //------------------------------------------------
    // Free Device Memory
    //------------------------------------------------

    CUDA_CHECK(
        cudaFree(d_input)
    );

    CUDA_CHECK(
        cudaFree(d_output)
    );


    return 0;
}
