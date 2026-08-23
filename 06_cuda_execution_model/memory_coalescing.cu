#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t error = (call);                         \
        if (error != cudaSuccess) {                         \
            std::cerr << "CUDA error: "                     \
                      << cudaGetErrorString(error)          \
                      << " at " << __FILE__                 \
                      << ':' << __LINE__                    \
                      << '\n';                              \
            std::exit(EXIT_FAILURE);                        \
        }                                                   \
    } while (0)


// ============================================================
// Kernel 1:
// 连续读取 + 连续写入
// ============================================================

__global__ void contiguous_copy(
    const float* input,
    float* output,
    int n) {

    const int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (idx < n) {
        output[idx] =
            input[idx];
    }
}


// ============================================================
// Kernel 2:
// 跨步读取 + 连续写入
// ============================================================

__global__ void strided_copy(
    const float* input,
    float* output,
    int rows,
    int cols) {

    const int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    const int n =
        rows * cols;

    if (idx < n) {

        const int row =
            idx / cols;

        const int col =
            idx % cols;

        const int input_idx =
            col * rows +
            row;

        output[idx] =
            input[input_idx];
    }
}


int main() {

    // ========================================================
    // 1. Problem size
    // ========================================================

    const int rows =
        4096;

    const int cols =
        4096;

    const int n =
        rows * cols;

    const std::size_t bytes =
        static_cast<std::size_t>(n)
        * sizeof(float);


    // ========================================================
    // 2. Host memory
    // ========================================================

    std::vector<float> h_input(n);

    std::vector<float> h_contiguous_output(n);

    std::vector<float> h_strided_output(n);


    // ========================================================
    // 3. Initialize input
    // ========================================================

    for (int idx = 0;
         idx < n;
         ++idx) {

        h_input[idx] =
            static_cast<float>(
                idx % 1000
            );
    }


    // ========================================================
    // 4. Print problem information
    // ========================================================

    std::cout
        << "rows = "
        << rows
        << '\n'
        << "cols = "
        << cols
        << '\n'
        << "element count = "
        << n
        << '\n'
        << "data size = "
        << static_cast<double>(bytes)
            / (1024.0 * 1024.0)
        << " MiB\n";


    // ========================================================
    // 5. Device pointers
    // ========================================================

    float* d_input =
        nullptr;

    float* d_contiguous_output =
        nullptr;

    float* d_strided_output =
        nullptr;


    // ========================================================
    // 6. Device memory allocation
    // ========================================================

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(
            &d_input
        ),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(
            &d_contiguous_output
        ),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(
            &d_strided_output
        ),
        bytes
    ));


    // ========================================================
    // 7. H2D
    // ========================================================

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));


    // ========================================================
    // 8. Launch configuration
    // ========================================================

    const int threads_per_block =
        256;

    const int blocks =
        (n + threads_per_block - 1)
        / threads_per_block;

    std::cout
        << "blocks = "
        << blocks
        << '\n'
        << "threads per block = "
        << threads_per_block
        << '\n';


    // ========================================================
    // 9. Correctness phase
    //
    // 先各运行一次。
    // 此时不测性能，只检查算得对不对。
    // ========================================================

    contiguous_copy<<<
        blocks,
        threads_per_block
    >>>(
        d_input,
        d_contiguous_output,
        n
    );

    CUDA_CHECK(cudaGetLastError());


    strided_copy<<<
        blocks,
        threads_per_block
    >>>(
        d_input,
        d_strided_output,
        rows,
        cols
    );

    CUDA_CHECK(cudaGetLastError());


    // 等待两个 Kernel 真正执行完成。
    CUDA_CHECK(cudaDeviceSynchronize());


    // ========================================================
    // 10. D2H for correctness checking
    // ========================================================

    CUDA_CHECK(cudaMemcpy(
        h_contiguous_output.data(),
        d_contiguous_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_strided_output.data(),
        d_strided_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));


    // ========================================================
    // 11. Validate contiguous kernel
    // ========================================================

    bool contiguous_correct =
        true;

    for (int idx = 0;
         idx < n;
         ++idx) {

        const float expected =
            h_input[idx];

        const float actual =
            h_contiguous_output[idx];

        if (std::fabs(
                expected - actual
            ) > 1e-6F) {

            contiguous_correct =
                false;

            std::cerr
                << "contiguous mismatch at idx = "
                << idx
                << ", expected = "
                << expected
                << ", actual = "
                << actual
                << '\n';

            break;
        }
    }


    // ========================================================
    // 12. Validate strided kernel
    // ========================================================

    bool strided_correct =
        true;

    for (int idx = 0;
         idx < n;
         ++idx) {

        const int row =
            idx / cols;

        const int col =
            idx % cols;

        const int input_idx =
            col * rows +
            row;

        const float expected =
            h_input[input_idx];

        const float actual =
            h_strided_output[idx];

        if (std::fabs(
                expected - actual
            ) > 1e-6F) {

            strided_correct =
                false;

            std::cerr
                << "strided mismatch at idx = "
                << idx
                << ", input_idx = "
                << input_idx
                << ", expected = "
                << expected
                << ", actual = "
                << actual
                << '\n';

            break;
        }
    }


    // ========================================================
    // 13. Print correctness
    // ========================================================

    std::cout
        << "contiguous correct = "
        << std::boolalpha
        << contiguous_correct
        << '\n'
        << "strided correct = "
        << strided_correct
        << '\n';


    // ========================================================
    // 14. Abort benchmark if correctness failed
    // ========================================================

    if (!contiguous_correct ||
        !strided_correct) {

        std::cerr
            << "Correctness check failed. "
            << "Benchmark aborted.\n";

        CUDA_CHECK(cudaFree(
            d_input
        ));

        CUDA_CHECK(cudaFree(
            d_contiguous_output
        ));

        CUDA_CHECK(cudaFree(
            d_strided_output
        ));

        return EXIT_FAILURE;
    }


    // ========================================================
    // 15. Benchmark configuration
    // ========================================================

    const int warmup_iterations =
        10;

    const int benchmark_iterations =
        100;


    // ========================================================
    // 16. Warmup
    // ========================================================

    for (int i = 0;
         i < warmup_iterations;
         ++i) {

        contiguous_copy<<<
            blocks,
            threads_per_block
        >>>(
            d_input,
            d_contiguous_output,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());


    for (int i = 0;
         i < warmup_iterations;
         ++i) {

        strided_copy<<<
            blocks,
            threads_per_block
        >>>(
            d_input,
            d_strided_output,
            rows,
            cols
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());


    // ========================================================
    // 17. Create CUDA Events
    // ========================================================

    cudaEvent_t start =
        nullptr;

    cudaEvent_t stop =
        nullptr;

    CUDA_CHECK(cudaEventCreate(
        &start
    ));

    CUDA_CHECK(cudaEventCreate(
        &stop
    ));


    // ========================================================
    // 18. Benchmark contiguous_copy
    // ========================================================

    CUDA_CHECK(cudaEventRecord(
        start
    ));

    for (int i = 0;
         i < benchmark_iterations;
         ++i) {

        contiguous_copy<<<
            blocks,
            threads_per_block
        >>>(
            d_input,
            d_contiguous_output,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(
        stop
    ));

    CUDA_CHECK(cudaEventSynchronize(
        stop
    ));

    float contiguous_total_ms =
        0.0F;

    CUDA_CHECK(cudaEventElapsedTime(
        &contiguous_total_ms,
        start,
        stop
    ));

    const float contiguous_average_ms =
        contiguous_total_ms
        / static_cast<float>(
            benchmark_iterations
        );


    // ========================================================
    // 19. Benchmark strided_copy
    // ========================================================

    CUDA_CHECK(cudaEventRecord(
        start
    ));

    for (int i = 0;
         i < benchmark_iterations;
         ++i) {

        strided_copy<<<
            blocks,
            threads_per_block
        >>>(
            d_input,
            d_strided_output,
            rows,
            cols
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(
        stop
    ));

    CUDA_CHECK(cudaEventSynchronize(
        stop
    ));

    float strided_total_ms =
        0.0F;

    CUDA_CHECK(cudaEventElapsedTime(
        &strided_total_ms,
        start,
        stop
    ));

    const float strided_average_ms =
        strided_total_ms
        / static_cast<float>(
            benchmark_iterations
        );


    // ========================================================
    // 20. Performance comparison
    // ========================================================

    const float slowdown =
        strided_average_ms
        / contiguous_average_ms;

    std::cout
        << "contiguous average = "
        << contiguous_average_ms
        << " ms\n"
        << "strided average = "
        << strided_average_ms
        << " ms\n"
        << "strided / contiguous = "
        << slowdown
        << "x\n";


    // ========================================================
    // 21. Cleanup
    // ========================================================

    CUDA_CHECK(cudaEventDestroy(
        start
    ));

    CUDA_CHECK(cudaEventDestroy(
        stop
    ));

    CUDA_CHECK(cudaFree(
        d_input
    ));

    CUDA_CHECK(cudaFree(
        d_contiguous_output
    ));

    CUDA_CHECK(cudaFree(
        d_strided_output
    ));


    return EXIT_SUCCESS;
}
