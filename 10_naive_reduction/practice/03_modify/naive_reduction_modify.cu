#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t error = (call);                 \
        if (error != cudaSuccess) {                 \
            std::cerr << "CUDA error: "             \
                      << cudaGetErrorString(error)   \
                      << " at " << __FILE__          \
                      << ":" << __LINE__             \
                      << '\n';                       \
            std::exit(EXIT_FAILURE);                \
        }                                           \
    } while (0)


// ============================================================
// Naive Reduction
//
// Modify version:
// 1. Global Memory 原地 Reduction
// 2. 单 Block
// 3. Interleaved Reduction
// 4. 将 Reduction 后的 d_data 拷回 Host 观察
// ============================================================
__global__ void reduce_naive(
    float* data,
    float* output,
    int n) {

    unsigned int tid =
        threadIdx.x;

    for (unsigned int stride = 1;
         stride < blockDim.x;
         stride *= 2) {

        if (tid % (2 * stride) == 0) {

            unsigned int other =
                tid + stride;

            if (tid < n && other < n) {
                data[tid] += data[other];
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[0] = data[0];
    }
}


// ============================================================
// CPU Reference
// ============================================================
float cpu_reduce(
    const std::vector<float>& input) {

    float sum = 0.0F;

    for (float value : input) {
        sum += value;
    }

    return sum;
}


int main() {
    constexpr int N = 8;
    constexpr int BLOCK_SIZE = 8;


    // --------------------------------------------------------
    // 1. Host input
    // --------------------------------------------------------
    std::vector<float> h_input(N);

    for (int i = 0; i < N; ++i) {
        h_input[i] =
            static_cast<float>(i + 1);
    }


    // --------------------------------------------------------
    // 2. CPU reference
    // --------------------------------------------------------
    float cpu_sum =
        cpu_reduce(h_input);


    // --------------------------------------------------------
    // 3. Device memory
    // --------------------------------------------------------
    float* d_data = nullptr;
    float* d_output = nullptr;

    size_t bytes =
        N * sizeof(float);

    CUDA_CHECK(cudaMalloc(
        &d_data,
        bytes));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        sizeof(float)));


    // --------------------------------------------------------
    // 4. H2D
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        d_data,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice));


    // --------------------------------------------------------
    // 5. Launch kernel
    // --------------------------------------------------------
    reduce_naive<<<1, BLOCK_SIZE>>>(
        d_data,
        d_output,
        N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // 6. D2H
    // --------------------------------------------------------
    float gpu_sum = 0.0F;

    std::vector<float> h_after(N);

    CUDA_CHECK(cudaMemcpy(
        &gpu_sum,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_after.data(),
        d_data,
        bytes,
        cudaMemcpyDeviceToHost));


    // --------------------------------------------------------
    // 7. Scalar correctness
    // --------------------------------------------------------
    bool correct =
        std::fabs(cpu_sum - gpu_sum) < 1e-5F;


    // --------------------------------------------------------
    // 8. Internal state correctness
    // --------------------------------------------------------
    const std::vector<float> expected_after{
        36.0F,
        2.0F,
        7.0F,
        4.0F,
        26.0F,
        6.0F,
        15.0F,
        8.0F
    };

    bool state_correct = true;

    for (int idx = 0;
         idx < N;
         ++idx) {

        const float expected =
            expected_after[idx];

        const float actual =
            h_after[idx];

        if (std::fabs(
                expected - actual
            ) > 1e-5F) {

            state_correct = false;

            std::cerr
                << "state mismatch at idx = "
                << idx
                << ", expected = "
                << expected
                << ", actual = "
                << actual
                << '\n';

            break;
        }
    }


    // --------------------------------------------------------
    // 9. Print result
    // --------------------------------------------------------
    std::cout << "input: ";

    for (float value : h_input) {
        std::cout << value << " ";
    }

    std::cout << "\n";


    std::cout << "data after reduction: ";

    for (float value : h_after) {
        std::cout << value << " ";
    }

    std::cout << "\n";


    std::cout << "CPU sum = "
              << cpu_sum << "\n";

    std::cout << "GPU sum = "
              << gpu_sum << "\n";

    std::cout << std::boolalpha
              << "correct = "
              << correct << "\n";

    std::cout << std::boolalpha
              << "state correct = "
              << state_correct
              << "\n";


    // --------------------------------------------------------
    // 10. Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
