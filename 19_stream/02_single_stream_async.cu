#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>


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
    // Pinned Host Memory
    //==================================================

    float* h_A =
        nullptr;

    float* h_B =
        nullptr;

    float* h_C =
        nullptr;


    CUDA_CHECK(
        cudaMallocHost(
            &h_A,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMallocHost(
            &h_B,
            bytes
        )
    );

    CUDA_CHECK(
        cudaMallocHost(
            &h_C,
            bytes
        )
    );


    for (
        int idx = 0;
        idx < N;
        ++idx
    ) {
        h_A[idx] =
            1.0F;

        h_B[idx] =
            2.0F;

        h_C[idx] =
            0.0F;
    }


    //==================================================
    // Device Memory
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
    // Create Stream
    //==================================================

    cudaStream_t stream;

    CUDA_CHECK(
        cudaStreamCreate(
            &stream
        )
    );


    //==================================================
    // Async H2D
    //==================================================

    CUDA_CHECK(
        cudaMemcpyAsync(
            d_A,
            h_A,
            bytes,
            cudaMemcpyHostToDevice,
            stream
        )
    );

    CUDA_CHECK(
        cudaMemcpyAsync(
            d_B,
            h_B,
            bytes,
            cudaMemcpyHostToDevice,
            stream
        )
    );


    //==================================================
    // Kernel in the same stream
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
        BLOCK_SIZE,
        0,
        stream
    >>>(
        d_A,
        d_B,
        d_C,
        N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //==================================================
    // Async D2H
    //==================================================

    CUDA_CHECK(
        cudaMemcpyAsync(
            h_C,
            d_C,
            bytes,
            cudaMemcpyDeviceToHost,
            stream
        )
    );


    //==================================================
    // Host needs result now
    //==================================================

    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
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
        if (
            std::fabs(
                h_C[idx]
                -
                3.0F
            )
            >
            1e-5F
        ) {
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


    CUDA_CHECK(
        cudaStreamDestroy(
            stream
        )
    );


    CUDA_CHECK(
        cudaFree(d_A)
    );

    CUDA_CHECK(
        cudaFree(d_B)
    );

    CUDA_CHECK(
        cudaFree(d_C)
    );


    CUDA_CHECK(
        cudaFreeHost(h_A)
    );

    CUDA_CHECK(
        cudaFreeHost(h_B)
    );

    CUDA_CHECK(
        cudaFreeHost(h_C)
    );


    return 0;
}
