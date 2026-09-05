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
void fill_one_kernel(
    float* data,
    int n
)
{
    const int idx =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;

    if (idx < n) {
        data[idx] =
            1.0F;
    }
}


__global__
void add_one_kernel(
    float* data,
    int n
)
{
    const int idx =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;

    if (idx < n) {
        data[idx]
            +=
            1.0F;
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
    // Pinned host output
    //==================================================

    float* h_output =
        nullptr;

    CUDA_CHECK(
        cudaMallocHost(
            &h_output,
            bytes
        )
    );


    //==================================================
    // Device
    //==================================================

    float* d_data =
        nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_data,
            bytes
        )
    );


    //==================================================
    // Two Streams
    //==================================================

    cudaStream_t stream0;
    cudaStream_t stream1;


    CUDA_CHECK(
        cudaStreamCreate(
            &stream0
        )
    );

    CUDA_CHECK(
        cudaStreamCreate(
            &stream1
        )
    );


    //==================================================
    // Event used for dependency, not timing
    //==================================================

    cudaEvent_t data_ready;

    CUDA_CHECK(
        cudaEventCreateWithFlags(
            &data_ready,
            cudaEventDisableTiming
        )
    );


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


    //==================================================
    // Stream0:
    // data = 1
    //==================================================

    fill_one_kernel<<<
        blocks,
        BLOCK_SIZE,
        0,
        stream0
    >>>(
        d_data,
        N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //==================================================
    // Record completion point
    //==================================================

    CUDA_CHECK(
        cudaEventRecord(
            data_ready,
            stream0
        )
    );


    //==================================================
    // Stream1:
    // wait until stream0 reaches event
    //==================================================

    CUDA_CHECK(
        cudaStreamWaitEvent(
            stream1,
            data_ready,
            0
        )
    );


    //==================================================
    // Then data += 1
    //==================================================

    add_one_kernel<<<
        blocks,
        BLOCK_SIZE,
        0,
        stream1
    >>>(
        d_data,
        N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //==================================================
    // D2H also in stream1
    //==================================================

    CUDA_CHECK(
        cudaMemcpyAsync(
            h_output,
            d_data,
            bytes,
            cudaMemcpyDeviceToHost,
            stream1
        )
    );


    CUDA_CHECK(
        cudaStreamSynchronize(
            stream1
        )
    );


    //==================================================
    // Verify
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
                h_output[idx]
                -
                2.0F
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
        cudaEventDestroy(
            data_ready
        )
    );

    CUDA_CHECK(
        cudaStreamDestroy(
            stream0
        )
    );

    CUDA_CHECK(
        cudaStreamDestroy(
            stream1
        )
    );

    CUDA_CHECK(
        cudaFree(d_data)
    );

    CUDA_CHECK(
        cudaFreeHost(h_output)
    );


    return 0;
}
