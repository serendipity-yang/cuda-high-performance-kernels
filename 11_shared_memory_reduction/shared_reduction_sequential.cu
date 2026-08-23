#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                           \
    do {                                           \
        cudaError_t error = (call);                \
        if (error != cudaSuccess) {                \
            std::cerr                              \
                << "CUDA error: "                  \
                << cudaGetErrorString(error)       \
                << " at "                          \
                << __FILE__                        \
                << ":"                             \
                << __LINE__                        \
                << '\n';                           \
            std::exit(EXIT_FAILURE);               \
        }                                          \
    } while (0)


// ============================================================
// Shared Memory Reduction V2
//
// Sequential Addressing
//
// 流程：
// Global Memory
//      ↓
// Shared Memory
//      ↓
// stride = blockDim.x / 2
//      ↓
// stride /= 2
//      ↓
// thread 0 得到 block sum
// ============================================================
__global__ void reduce_shared_sequential(
    const float* input,
    float* output,
    int n) {

    // --------------------------------------------------------
    // Dynamic Shared Memory
    // --------------------------------------------------------
    extern __shared__ float sdata[];


    // --------------------------------------------------------
    // Local thread index
    // --------------------------------------------------------
    unsigned int tid =
        threadIdx.x;


    // --------------------------------------------------------
    // 1. Global Memory -> Shared Memory
    // --------------------------------------------------------
    if (tid < n) {

        sdata[tid] =
            input[tid];

    } else {

        sdata[tid] =
            0.0F;
    }


    // 等待整个 Block 完成数据加载
    __syncthreads();


    // --------------------------------------------------------
    // 2. Sequential Addressing Reduction
    //
    // 例如 blockDim.x = 8:
    //
    // stride:
    //
    // 4
    // 2
    // 1
    // --------------------------------------------------------
    for (unsigned int stride =
             blockDim.x / 2;
         stride > 0;
         stride /= 2) {


        // 前 stride 个线程参与计算
        if (tid < stride) {

            sdata[tid] +=
                sdata[tid + stride];
        }


        // 等待这一轮 Reduction 全部结束
        __syncthreads();
    }


    // --------------------------------------------------------
    // 3. Write block result
    // --------------------------------------------------------
    if (tid == 0) {

        output[0] =
            sdata[0];
    }
}


// ============================================================
// CPU Reference
// ============================================================
float cpu_reduce(
    const std::vector<float>& input) {

    float sum =
        0.0F;


    for (float value : input) {

        sum +=
            value;
    }


    return sum;
}


// ============================================================
// main
// ============================================================
int main() {

    constexpr int N =
        8;

    constexpr int BLOCK_SIZE =
        8;


    // --------------------------------------------------------
    // 1. Host input
    // --------------------------------------------------------
    std::vector<float> h_input(N);


    for (int i = 0;
         i < N;
         ++i) {

        h_input[i] =
            static_cast<float>(
                i + 1);
    }


    // --------------------------------------------------------
    // 2. CPU reference
    // --------------------------------------------------------
    float cpu_sum =
        cpu_reduce(
            h_input);


    // --------------------------------------------------------
    // 3. Device pointers
    // --------------------------------------------------------
    float* d_input =
        nullptr;

    float* d_output =
        nullptr;


    size_t input_bytes =
        N * sizeof(float);


    // --------------------------------------------------------
    // 4. Device allocation
    // --------------------------------------------------------
    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            input_bytes));


    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            sizeof(float)));


    // --------------------------------------------------------
    // 5. H2D
    // --------------------------------------------------------
    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            input_bytes,
            cudaMemcpyHostToDevice));


    // --------------------------------------------------------
    // 6. Shared Memory size
    // --------------------------------------------------------
    size_t shared_bytes =
        BLOCK_SIZE *
        sizeof(float);


    // --------------------------------------------------------
    // 7. Kernel launch
    // --------------------------------------------------------
    reduce_shared_sequential<<<
        1,
        BLOCK_SIZE,
        shared_bytes>>>(
            d_input,
            d_output,
            N);


    CUDA_CHECK(
        cudaGetLastError());


    CUDA_CHECK(
        cudaDeviceSynchronize());


    // --------------------------------------------------------
    // 8. D2H
    // --------------------------------------------------------
    float gpu_sum =
        0.0F;


    CUDA_CHECK(
        cudaMemcpy(
            &gpu_sum,
            d_output,
            sizeof(float),
            cudaMemcpyDeviceToHost));


    // --------------------------------------------------------
    // 9. Correctness
    // --------------------------------------------------------
    bool correct =
        std::fabs(
            cpu_sum -
            gpu_sum)
        < 1e-5F;


    // --------------------------------------------------------
    // 10. Print
    // --------------------------------------------------------
    std::cout
        << "input: ";


    for (float value : h_input) {

        std::cout
            << value
            << " ";
    }


    std::cout
        << '\n';


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


    // --------------------------------------------------------
    // 11. Cleanup
    // --------------------------------------------------------
    CUDA_CHECK(
        cudaFree(
            d_input));


    CUDA_CHECK(
        cudaFree(
            d_output));


    return 0;
}
