#include <cuda_runtime.h>

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


__global__
void vector_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int n
)
{
    const int idx =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;

    if (idx < n) {
        C[idx] =
            A[idx]
            +
            B[idx];
    }
}


int main()
{
    constexpr int N =
        1 << 20;

    constexpr int BLOCK_SIZE =
        256;

    const std::size_t bytes =
        static_cast<std::size_t>(N)
        *
        sizeof(float);


    //==================================================
    // Host pageable memory
    //==================================================

    std::vector<float> h_A(
        N,
        1.0F
    );

    std::vector<float> h_B(
        N,
        2.0F
    );

    std::vector<float> h_C(
        N,
        0.0F
    );


    //==================================================
    // Device memory
    //==================================================

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            bytes
        )
    );


    //==================================================
    // Host -> Device
    //==================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    //==================================================
    // Kernel
    //==================================================

    const int blocks =
        (
            N
            +
            BLOCK_SIZE
            -
            1
        )
        /
        BLOCK_SIZE;


    vector_add_kernel<<<
        blocks,
        BLOCK_SIZE
    >>>(
        d_A,
        d_B,
        d_C,
        N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    //==================================================
    // Device -> Host
    //==================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_C.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //==================================================
    // Correctness
    //==================================================

    bool correct =
        true;

    for (
        int idx = 0;
        idx < N;
        ++idx
    ) {
        const float expected =
            3.0F;

        const float actual =
            h_C[idx];

        if (
            std::fabs(
                actual
                -
                expected
            )
            >
            1e-5F
        ) {
            correct =
                false;

            std::cerr
                << "Mismatch at idx = "
                << idx
                << '\n';

            break;
        }
    }


    std::cout
        << std::boolalpha
        << "correct = "
        << correct
        << '\n';


    CUDA_CHECK(
        cudaFree(d_A)
    );

    CUDA_CHECK(
        cudaFree(d_B)
    );

    CUDA_CHECK(
        cudaFree(d_C)
    );


    return 0;
}
