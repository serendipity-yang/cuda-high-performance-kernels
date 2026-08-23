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
// Version 1
// Interleaved Reduction
// ============================================================

__global__ void reduce_interleaved(
    const float* input,
    float* partial_sum,
    int n) {

    extern __shared__ float sdata[];

    const unsigned int tid =
        threadIdx.x;

    const unsigned int global_idx =
        blockIdx.x *
            blockDim.x *
            2 +
        tid;


    // --------------------------------------------------------
    // 每个线程先负责最多两个元素
    // --------------------------------------------------------

    float value = 0.0F;

    if (global_idx < n) {

        value +=
            input[global_idx];
    }

    const unsigned int second_idx =
        global_idx +
        blockDim.x;

    if (second_idx < n) {

        value +=
            input[second_idx];
    }


    // --------------------------------------------------------
    // 写入 shared memory
    // --------------------------------------------------------

    sdata[tid] =
        value;

    __syncthreads();


    // --------------------------------------------------------
    // Interleaved Reduction
    // --------------------------------------------------------

    for (unsigned int stride = 1;
         stride < blockDim.x;
         stride *= 2) {

        if (tid % (2 * stride) == 0) {

            sdata[tid] +=
                sdata[tid + stride];
        }

        __syncthreads();
    }


    // --------------------------------------------------------
    // 每个 block 输出一个 partial sum
    // --------------------------------------------------------

    if (tid == 0) {

        partial_sum[blockIdx.x] =
            sdata[0];
    }
}


// ============================================================
// Version 2
// Sequential / Continuous Reduction
// ============================================================

__global__ void reduce_sequential(
    const float* input,
    float* partial_sum,
    int n) {

    extern __shared__ float sdata[];

    const unsigned int tid =
        threadIdx.x;

    const unsigned int global_idx =
        blockIdx.x *
            blockDim.x *
            2 +
        tid;


    float value = 0.0F;

    if (global_idx < n) {

        value +=
            input[global_idx];
    }

    const unsigned int second_idx =
        global_idx +
        blockDim.x;

    if (second_idx < n) {

        value +=
            input[second_idx];
    }


    sdata[tid] =
        value;

    __syncthreads();


    // --------------------------------------------------------
    // Sequential Reduction
    // --------------------------------------------------------

    for (unsigned int stride =
             blockDim.x / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {

            sdata[tid] +=
                sdata[tid + stride];
        }

        __syncthreads();
    }


    if (tid == 0) {

        partial_sum[blockIdx.x] =
            sdata[0];
    }
}


// ============================================================
// CPU 求 partial sums 的最终结果
// ============================================================

double collect_partial_sum(
    const std::vector<float>& partial) {

    double sum = 0.0;

    for (float value : partial) {

        sum +=
            static_cast<double>(value);
    }

    return sum;
}


// ============================================================
// benchmark interleaved
// ============================================================

float benchmark_interleaved(
    const float* d_input,
    float* d_partial,
    int n,
    int blocks,
    int threads,
    int repeat) {

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start));

    CUDA_CHECK(
        cudaEventCreate(&stop));


    const std::size_t shared_bytes =
        threads *
        sizeof(float);


    // warmup
    for (int i = 0;
         i < 10;
         ++i) {

        reduce_interleaved<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }

    CUDA_CHECK(
        cudaDeviceSynchronize());


    CUDA_CHECK(
        cudaEventRecord(start));


    for (int i = 0;
         i < repeat;
         ++i) {

        reduce_interleaved<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }


    CUDA_CHECK(
        cudaEventRecord(stop));

    CUDA_CHECK(
        cudaEventSynchronize(stop));


    float elapsed_ms = 0.0F;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));


    CUDA_CHECK(
        cudaEventDestroy(start));

    CUDA_CHECK(
        cudaEventDestroy(stop));


    return elapsed_ms /
           static_cast<float>(repeat);
}


// ============================================================
// benchmark sequential
// ============================================================

float benchmark_sequential(
    const float* d_input,
    float* d_partial,
    int n,
    int blocks,
    int threads,
    int repeat) {

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start));

    CUDA_CHECK(
        cudaEventCreate(&stop));


    const std::size_t shared_bytes =
        threads *
        sizeof(float);


    for (int i = 0;
         i < 10;
         ++i) {

        reduce_sequential<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }

    CUDA_CHECK(
        cudaDeviceSynchronize());


    CUDA_CHECK(
        cudaEventRecord(start));


    for (int i = 0;
         i < repeat;
         ++i) {

        reduce_sequential<<<
            blocks,
            threads,
            shared_bytes>>>(
                d_input,
                d_partial,
                n);
    }


    CUDA_CHECK(
        cudaEventRecord(stop));

    CUDA_CHECK(
        cudaEventSynchronize(stop));


    float elapsed_ms = 0.0F;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));


    CUDA_CHECK(
        cudaEventDestroy(start));

    CUDA_CHECK(
        cudaEventDestroy(stop));


    return elapsed_ms /
           static_cast<float>(repeat);
}


// ============================================================
// main
// ============================================================

int main() {

    constexpr int n =
        1 << 22;

    constexpr int threads =
        256;

    constexpr int repeat =
        100;


    // --------------------------------------------------------
    // 为什么除以 threads * 2？
    //
    // 因为每个线程最多读取两个元素
    // --------------------------------------------------------

    const int blocks =
        (n +
         threads * 2 -
         1) /
        (threads * 2);


    std::cout
        << "n       = "
        << n
        << '\n';

    std::cout
        << "threads = "
        << threads
        << '\n';

    std::cout
        << "blocks  = "
        << blocks
        << '\n';


    // ========================================================
    // Host input
    // ========================================================

    std::vector<float> h_input(
        n,
        1.0F);


    double cpu_sum = 0.0;

    for (float value : h_input) {

        cpu_sum +=
            static_cast<double>(value);
    }


    // ========================================================
    // Device memory
    // ========================================================

    float* d_input =
        nullptr;

    float* d_partial =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n *
            sizeof(float)));

    CUDA_CHECK(
        cudaMalloc(
            &d_partial,
            blocks *
            sizeof(float)));


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n *
            sizeof(float),
            cudaMemcpyHostToDevice));


    const std::size_t shared_bytes =
        threads *
        sizeof(float);


    // ========================================================
    // Interleaved result
    // ========================================================

    reduce_interleaved<<<
        blocks,
        threads,
        shared_bytes>>>(
            d_input,
            d_partial,
            n);

    CUDA_CHECK(
        cudaGetLastError());

    CUDA_CHECK(
        cudaDeviceSynchronize());


    std::vector<float>
        h_partial_interleaved(
            blocks);


    CUDA_CHECK(
        cudaMemcpy(
            h_partial_interleaved.data(),
            d_partial,
            blocks *
            sizeof(float),
            cudaMemcpyDeviceToHost));


    const double interleaved_sum =
        collect_partial_sum(
            h_partial_interleaved);


    // ========================================================
    // Sequential result
    // ========================================================

    reduce_sequential<<<
        blocks,
        threads,
        shared_bytes>>>(
            d_input,
            d_partial,
            n);

    CUDA_CHECK(
        cudaGetLastError());

    CUDA_CHECK(
        cudaDeviceSynchronize());


    std::vector<float>
        h_partial_sequential(
            blocks);


    CUDA_CHECK(
        cudaMemcpy(
            h_partial_sequential.data(),
            d_partial,
            blocks *
            sizeof(float),
            cudaMemcpyDeviceToHost));


    const double sequential_sum =
        collect_partial_sum(
            h_partial_sequential);


    // ========================================================
    // Timing
    // ========================================================

    const float interleaved_ms =
        benchmark_interleaved(
            d_input,
            d_partial,
            n,
            blocks,
            threads,
            repeat);


    const float sequential_ms =
        benchmark_sequential(
            d_input,
            d_partial,
            n,
            blocks,
            threads,
            repeat);


    // ========================================================
    // Results
    // ========================================================

    std::cout
        << "\nCPU sum         = "
        << cpu_sum
        << '\n';

    std::cout
        << "interleaved sum = "
        << interleaved_sum
        << '\n';

    std::cout
        << "sequential sum  = "
        << sequential_sum
        << '\n';


    const bool interleaved_correct =
        std::fabs(
            interleaved_sum -
            cpu_sum) <
        1e-3;

    const bool sequential_correct =
        std::fabs(
            sequential_sum -
            cpu_sum) <
        1e-3;


    std::cout
        << std::boolalpha;

    std::cout
        << "\ninterleaved correct = "
        << interleaved_correct
        << '\n';

    std::cout
        << "sequential correct  = "
        << sequential_correct
        << '\n';


    std::cout
        << "\ninterleaved average = "
        << interleaved_ms
        << " ms\n";

    std::cout
        << "sequential average  = "
        << sequential_ms
        << " ms\n";


    std::cout
        << "interleaved / sequential = "
        << interleaved_ms /
               sequential_ms
        << "x\n";


    CUDA_CHECK(
        cudaFree(d_input));

    CUDA_CHECK(
        cudaFree(d_partial));


    return 0;
}
