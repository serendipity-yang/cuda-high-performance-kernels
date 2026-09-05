#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                  \
    do {                                                  \
        cudaError_t error = (call);                       \
        if (error != cudaSuccess) {                       \
            std::cerr                                     \
                << "CUDA error: "                         \
                << cudaGetErrorString(error)              \
                << " at "                                 \
                << __FILE__                               \
                << ":"                                    \
                << __LINE__                               \
                << '\n';                                  \
            std::exit(EXIT_FAILURE);                      \
        }                                                 \
    } while (0)


#define CUBLAS_CHECK(call)                                \
    do {                                                  \
        cublasStatus_t status = (call);                   \
        if (status != CUBLAS_STATUS_SUCCESS) {            \
            std::cerr                                     \
                << "cuBLAS error at "                     \
                << __FILE__                               \
                << ":"                                    \
                << __LINE__                               \
                << '\n';                                  \
            std::exit(EXIT_FAILURE);                      \
        }                                                 \
    } while (0)


int main()
{
    //====================================================
    // A: M x K
    //
    // B: K x N
    //
    // C: M x N
    //====================================================

    const int M =
        2;

    const int K =
        3;

    const int N =
        4;


    //====================================================
    // Host matrices
    //====================================================

    std::vector<float> h_A = {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F
    };

    std::vector<float> h_B = {
         1.0F,  2.0F,  3.0F,  4.0F,
         5.0F,  6.0F,  7.0F,  8.0F,
         9.0F, 10.0F, 11.0F, 12.0F
    };

    std::vector<float> h_C(
        M * N,
        0.0F
    );


    //====================================================
    // Expected result:
    //
    // 58   64
    // 139 154
    //====================================================

    const std::vector<float> expected = {
         38.0F,
         44.0F,
         50.0F,
         56.0F,

         83.0F,
         98.0F,
        113.0F,
        128.0F
    };

    //====================================================
    // Device pointers
    //====================================================

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;


    const std::size_t bytes_A =
        M * K * sizeof(float);

    const std::size_t bytes_B =
        K * N * sizeof(float);

    const std::size_t bytes_C =
        M * N * sizeof(float);


    //====================================================
    // Allocate GPU memory
    //====================================================

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            bytes_A
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            bytes_B
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            bytes_C
        )
    );


    //====================================================
    // H2D
    //====================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes_A,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytes_B,
            cudaMemcpyHostToDevice
        )
    );


    //====================================================
    // Create cuBLAS handle
    //====================================================

    cublasHandle_t handle;

    CUBLAS_CHECK(
        cublasCreate(
            &handle
        )
    );


    //====================================================
    // C = alpha * A * B + beta * C
    //====================================================

    const float alpha =
        1.0F;

    const float beta =
        0.0F;


    //====================================================
    // Row-Major trick:
    //
    // C = A B
    //
    // =>
    //
    // C^T = B^T A^T
    //
    // Therefore:
    //
    // m = N
    // n = M
    // k = K
    //
    // first matrix  = d_B
    // second matrix = d_A
    //====================================================

    CUBLAS_CHECK(
        cublasSgemm(
            handle,

            CUBLAS_OP_N,
            CUBLAS_OP_N,

            N,
            M,
            K,

            &alpha,

            d_B,
            N,

            d_A,
            K,

            &beta,

            d_C,
            N
        )
    );


    //====================================================
    // Wait for GPU
    //====================================================

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    //====================================================
    // D2H
    //====================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );


    //====================================================
    // Print C
    //====================================================

    std::cout
        << "C =\n";


    for (
        int row = 0;
        row < M;
        ++row
    )
    {
        for (
            int col = 0;
            col < N;
            ++col
        )
        {
            std::cout
                << h_C[
                    row * N
                    +
                    col
                ]
                << '\t';
        }

        std::cout
            << '\n';
    }


    //====================================================
    // Correctness
    //====================================================

    bool correct =
        true;


    for (
        int idx = 0;
        idx < M * N;
        ++idx
    )
    {
        if (
            std::fabs(
                h_C[idx]
                -
                expected[idx]
            )
            >
            1e-5F
        )
        {
            correct =
                false;

            break;
        }
    }


    std::cout
        << std::boolalpha
        << "correct = "
        << correct
        << '\n';


    //====================================================
    // Cleanup
    //====================================================

    CUBLAS_CHECK(
        cublasDestroy(
            handle
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_A
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_B
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_C
        )
    );


    return 0;
}
