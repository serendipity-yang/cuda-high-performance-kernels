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

__global__
void transpose_tiled(
    const float* input,
    float* output,
    int width,
    int height) {

    __shared__
    float tile[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * TILE_DIM + tx;
    const int y = blockIdx.y * TILE_DIM + ty;
    
    if (x < width && y < height) {

        tile[ty][tx] = input[y * width + x];
    }

    __syncthreads();

    const int out_x = 
        blockIdx.y * TILE_DIM + tx;
    
    const int out_y =
        blockIdx.x * TILE_DIM + ty;

    if (out_x < height && out_y < width) {

        output[out_y * height + out_x] = tile[tx][ty];
    }
}

int main() {

    constexpr int width = 64;
    constexpr int height = 64;

    const int num_elements = width * height;
    
    const size_t bytes = num_elements * sizeof(float);

    std::vector<float> h_input(num_elements);

    for (int i = 0; i < num_elements; ++i) {

        h_input[i] = static_cast<float>(i);
    }

    std::vector<float> h_output(num_elements,0.0F);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice));

    dim3 block(
        TILE_DIM,
        TILE_DIM);

    dim3 grid(
        (width + TILE_DIM - 1)
        / TILE_DIM,
        (height + TILE_DIM - 1)
         / TILE_DIM);

    transpose_tiled
    <<<grid, block>>>(
        d_input,
        d_output,
        width,
        height);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
            h_output.data(),
            d_output,
            bytes,
            cudaMemcpyDeviceToHost));

    bool correct = true;

    for (int input_row = 0;
        input_row < height;
        ++input_row) {

        for(int input_col = 0;
            input_col < width;
            ++input_col) {

            const float expected =
                h_input[input_row * width + input_col];

            const float actual = 
                h_output[input_col * height + input_row];

            if (std::fabs(expected - actual) > 1e-6F) {

                correct = false;

                break;
            }
        }

        if (!correct) {
            break;
        }
    }

    std::cout << "correct = "
              << std::boolalpha
              << correct
              << '\n';
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
