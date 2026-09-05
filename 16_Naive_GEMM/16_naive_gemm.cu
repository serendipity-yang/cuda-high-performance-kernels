#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess) {                           \
            std::cerr                                         \
                << "CUDA error: "                             \
                << cudaGetErrorString(error)                  \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << '\n';                                      \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


//============================================================
// Naive GEMM
//
// A: M x K
// B: K x N
// C: M x N
//
// One Thread computes one C[row][col]
//============================================================

__global__
void naive_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{
    const int row =
        blockIdx.y
        *
        blockDim.y
        +
        threadIdx.y;


    const int col =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    if (
        row >= M
        ||
        col >= N
    )
    {
        return;
    }


    float sum =
        0.0F;


    for (
        int k = 0;
        k < K;
        ++k
    )
    {
        const float a =
            A[
                row * K
                +
                k
            ];


        const float b =
            B[
                k * N
                +
                col
            ];


        sum +=
            a * b;
    }


    C[
        row * N
        +
        col
    ]
        =
        sum;
}


//============================================================
// CPU Reference
//============================================================

void gemm_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
)
{
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
            float sum =
                0.0F;


            for (
                int k = 0;
                k < K;
                ++k
            )
            {
                sum +=
                    A[
                        row * K
                        +
                        k
                    ]
                    *
                    B[
                        k * N
                        +
                        col
                    ];
            }


            C[
                row * N
                +
                col
            ]
                =
                sum;
        }
    }
}


//============================================================
// main
//============================================================

int main()
{
    //========================================================
    // Matrix Shapes
    //
    // A: M x K
    // B: K x N
    // C: M x N
    //========================================================

    const int M =
        4;

    const int K =
        3;

    const int N =
        5;


    const std::size_t bytes_A =
        static_cast<std::size_t>(
            M
        )
        *
        K
        *
        sizeof(float);


    const std::size_t bytes_B =
        static_cast<std::size_t>(
            K
        )
        *
        N
        *
        sizeof(float);


    const std::size_t bytes_C =
        static_cast<std::size_t>(
            M
        )
        *
        N
        *
        sizeof(float);


    //========================================================
    // Host
    //========================================================

    std::vector<float>
        h_A(
            M * K
        );


    std::vector<float>
        h_B(
            K * N
        );


    std::vector<float>
        h_cpu_C(
            M * N
        );


    std::vector<float>
        h_gpu_C(
            M * N
        );


    //========================================================
    // Initialize
    //========================================================

    for (
        int idx = 0;
        idx < M * K;
        ++idx
    )
    {
        h_A[idx] =
            static_cast<float>(
                idx + 1
            );
    }


    for (
        int idx = 0;
        idx < K * N;
        ++idx
    )
    {
        h_B[idx] =
            1.0F;
    }


    //========================================================
    // CPU Reference
    //========================================================

    gemm_cpu(
        h_A,
        h_B,
        h_cpu_C,
        M,
        N,
        K
    );


    //========================================================
    // Device
    //========================================================

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;


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


    //========================================================
    // H2D
    //========================================================

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


    //========================================================
    // Block / Grid
    //========================================================

    dim3 block(
        16,
        16
    );


    dim3 grid(
        (
            N
            +
            block.x
            -
            1
        )
        /
        block.x,

        (
            M
            +
            block.y
            -
            1
        )
        /
        block.y
    );


    //========================================================
    // Kernel
    //========================================================

    naive_gemm_kernel<<<
        grid,
        block
    >>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    //========================================================
    // D2H
    //========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_gpu_C.data(),
            d_C,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );


    //========================================================
    // Correctness
    //========================================================

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
                h_cpu_C[idx]
                -
                h_gpu_C[idx]
            )
            >
            1e-5F
        )
        {
            correct =
                false;


            std::cerr
                << "Mismatch at idx = "
                << idx
                << ", CPU = "
                << h_cpu_C[idx]
                << ", GPU = "
                << h_gpu_C[idx]
                << '\n';


            break;
        }
    }


    //========================================================
    // Print
    //========================================================

    std::cout
        << std::boolalpha;


    std::cout
        << "correct = "
        << correct
        << '\n';


    std::cout
        << "\nGPU C:\n";


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
                << h_gpu_C[
                    row * N
                    +
                    col
                ]
                << ' ';
        }


        std::cout
            << '\n';
    }


    //========================================================
    // Free
    //========================================================

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
