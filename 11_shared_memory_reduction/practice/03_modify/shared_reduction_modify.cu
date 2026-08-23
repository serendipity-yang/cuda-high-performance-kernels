#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                           \
    do {                                           \
        cudaError_t error = (call);                \
        if (error != cudaSuccess) {                \
            std::cerr << "CUDA error: "            \
                      << cudaGetErrorString(error)  \
                      << " at "                     \
                      << __FILE__                   \
                      << ":"                        \
                      << __LINE__                   \
                      << '\n';                      \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)


__global__ void reduce_shared(
    const float* input,
    float* output,
    int n) {

    extern __shared__ float shared[];

    unsigned int tid =
        threadIdx.x;


    // ========================================================
    // Global Memory -> Shared Memory
    //
    // 有效线程：读取 input
    // 多出来的线程：填 0
    // ========================================================
    if (tid < n) {
        shared[tid] =
            input[tid];
    } else {
        shared[tid] =
            0.0F;
    }

    __syncthreads();


    // ========================================================
    // Shared Memory Reduction
    // ========================================================
    for (unsigned int stride = 1;
         stride < blockDim.x;
         stride *= 2) {

        if (tid % (2 * stride) == 0) {

            unsigned int other =
                tid + stride;

            if (other < blockDim.x) {

                shared[tid] +=
                    shared[other];
            }
        }

        __syncthreads();
    }


    // ========================================================
    // Final result
    // ========================================================
    if (tid == 0) {
        output[0] =
            shared[0];
    }
}


float cpu_reduce(
    const std::vector<float>& input) {

    float sum = 0.0F;

    for (float value : input) {
        sum += value;
    }

    return sum;
}


int main() {

    // ========================================================
    // Modification:
    //
    // N = 13
    // BLOCK_SIZE = 16
    //
    // 后三个线程没有对应输入数据
    // ========================================================
    constexpr int N = 13;
    constexpr int BLOCK_SIZE = 16;


    std::vector<float> h_input(N);

    for (int i = 0; i < N; ++i) {

        h_input[i] =
            static_cast<float>(i + 1);
    }


    float cpu_sum =
        cpu_reduce(h_input);


    float* d_input = nullptr;
    float* d_output = nullptr;


    size_t input_bytes =
        N * sizeof(float);


    CUDA_CHECK(cudaMalloc(
        &d_input,
        input_bytes));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        sizeof(float)));


    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        input_bytes,
        cudaMemcpyHostToDevice));


    size_t shared_bytes =
        BLOCK_SIZE * sizeof(float);


    reduce_shared<<<
        1,
        BLOCK_SIZE,
        shared_bytes>>>(
            d_input,
            d_output,
            N);


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    float gpu_sum = 0.0F;


    CUDA_CHECK(cudaMemcpy(
        &gpu_sum,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost));


    bool correct =
        std::fabs(cpu_sum - gpu_sum)
        < 1e-5F;


    std::cout
        << "N = "
        << N
        << '\n';

    std::cout
        << "BLOCK_SIZE = "
        << BLOCK_SIZE
        << '\n';

    std::cout
        << "shared bytes = "
        << shared_bytes
        << '\n';


    std::cout << "input: ";

    for (float value : h_input) {

        std::cout
            << value
            << " ";
    }

    std::cout << '\n';


    std::cout
        << "CPU sum = "
        << cpu_sum
        << '\n';

    std::cout
        << "GPU sum = "
        << gpu_sum
        << '\n';

    std::cout
        << std::boolalpha
        << "correct = "
        << correct
        << '\n';


    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));


    return 0;
}
