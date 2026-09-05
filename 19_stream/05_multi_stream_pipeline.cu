#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
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
    constexpr int NUM_STREAMS =
        2;

    constexpr int NUM_BATCHES =
        8;

    constexpr int CHUNK_N =
        1 << 20;

    constexpr int BLOCK_SIZE =
        256;


    const std::size_t chunk_bytes =
        static_cast<std::size_t>(CHUNK_N)
        *
        sizeof(float);


    const std::size_t total_elements =
        static_cast<std::size_t>(
            NUM_BATCHES
        )
        *
        CHUNK_N;


    const std::size_t total_bytes =
        total_elements
        *
        sizeof(float);


    //==================================================
    // Print device concurrency capability
    //==================================================

    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );


    std::cout
        << "GPU: "
        << prop.name
        << '\n';

    std::cout
        << "asyncEngineCount = "
        << prop.asyncEngineCount
        << '\n';

    std::cout
        << "concurrentKernels = "
        << prop.concurrentKernels
        << "\n\n";


    //==================================================
    // Full pinned host dataset
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
            total_bytes
        )
    );

    CUDA_CHECK(
        cudaMallocHost(
            &h_B,
            total_bytes
        )
    );

    CUDA_CHECK(
        cudaMallocHost(
            &h_C,
            total_bytes
        )
    );


    for (
        std::size_t idx = 0;
        idx < total_elements;
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
    // Two device slots
    //==================================================

    float* d_A[NUM_STREAMS]{};
    float* d_B[NUM_STREAMS]{};
    float* d_C[NUM_STREAMS]{};


    for (
        int s = 0;
        s < NUM_STREAMS;
        ++s
    ) {
        CUDA_CHECK(
            cudaMalloc(
                &d_A[s],
                chunk_bytes
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_B[s],
                chunk_bytes
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_C[s],
                chunk_bytes
            )
        );
    }


    //==================================================
    // Streams
    //==================================================

    cudaStream_t streams[NUM_STREAMS];


    for (
        int s = 0;
        s < NUM_STREAMS;
        ++s
    ) {
        CUDA_CHECK(
            cudaStreamCreate(
                &streams[s]
            )
        );
    }


    const int blocks =
        (
            CHUNK_N
            +
            BLOCK_SIZE
            -
            1
        )
        /
        BLOCK_SIZE;


    //==================================================
    // 1. Single Stream Baseline
    //
    // Use stream0 only.
    // Same pinned memory.
    // Same async API.
    //
    // Difference is only scheduling.
    //==================================================

    auto single_start =
        std::chrono::steady_clock::now();


    for (
        int batch = 0;
        batch < NUM_BATCHES;
        ++batch
    ) {
        const std::size_t offset =
            static_cast<std::size_t>(
                batch
            )
            *
            CHUNK_N;


        CUDA_CHECK(
            cudaMemcpyAsync(
                d_A[0],
                h_A + offset,
                chunk_bytes,
                cudaMemcpyHostToDevice,
                streams[0]
            )
        );


        CUDA_CHECK(
            cudaMemcpyAsync(
                d_B[0],
                h_B + offset,
                chunk_bytes,
                cudaMemcpyHostToDevice,
                streams[0]
            )
        );


        vector_add_kernel<<<
            blocks,
            BLOCK_SIZE,
            0,
            streams[0]
        >>>(
            d_A[0],
            d_B[0],
            d_C[0],
            CHUNK_N
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaMemcpyAsync(
                h_C + offset,
                d_C[0],
                chunk_bytes,
                cudaMemcpyDeviceToHost,
                streams[0]
            )
        );
    }


    CUDA_CHECK(
        cudaStreamSynchronize(
            streams[0]
        )
    );


    auto single_stop =
        std::chrono::steady_clock::now();


    const double single_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            single_stop
            -
            single_start
        ).count();


    //==================================================
    // 2. Multi Stream Pipeline
    //==================================================

    auto multi_start =
        std::chrono::steady_clock::now();


    for (
        int batch = 0;
        batch < NUM_BATCHES;
        ++batch
    ) {
        const int stream_id =
            batch
            %
            NUM_STREAMS;


        const std::size_t offset =
            static_cast<std::size_t>(
                batch
            )
            *
            CHUNK_N;


        CUDA_CHECK(
            cudaMemcpyAsync(
                d_A[stream_id],
                h_A + offset,
                chunk_bytes,
                cudaMemcpyHostToDevice,
                streams[stream_id]
            )
        );


        CUDA_CHECK(
            cudaMemcpyAsync(
                d_B[stream_id],
                h_B + offset,
                chunk_bytes,
                cudaMemcpyHostToDevice,
                streams[stream_id]
            )
        );


        vector_add_kernel<<<
            blocks,
            BLOCK_SIZE,
            0,
            streams[stream_id]
        >>>(
            d_A[stream_id],
            d_B[stream_id],
            d_C[stream_id],
            CHUNK_N
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaMemcpyAsync(
                h_C + offset,
                d_C[stream_id],
                chunk_bytes,
                cudaMemcpyDeviceToHost,
                streams[stream_id]
            )
        );
    }


    for (
        int s = 0;
        s < NUM_STREAMS;
        ++s
    ) {
        CUDA_CHECK(
            cudaStreamSynchronize(
                streams[s]
            )
        );
    }


    auto multi_stop =
        std::chrono::steady_clock::now();


    const double multi_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            multi_stop
            -
            multi_start
        ).count();


    //==================================================
    // Correctness
    //==================================================

    bool correct =
        true;


    for (
        std::size_t idx = 0;
        idx < total_elements;
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

            std::cerr
                << "Mismatch at idx = "
                << idx
                << '\n';

            break;
        }
    }


    //==================================================
    // Print
    //==================================================

    std::cout
        << std::fixed
        << std::setprecision(3);


    std::cout
        << "correct = "
        << std::boolalpha
        << correct
        << "\n\n";


    std::cout
        << "Single Stream total = "
        << single_ms
        << " ms\n";


    std::cout
        << "Multi Stream total  = "
        << multi_ms
        << " ms\n";


    std::cout
        << "Speedup = "
        << single_ms / multi_ms
        << "x\n";


    //==================================================
    // Cleanup
    //==================================================

    for (
        int s = 0;
        s < NUM_STREAMS;
        ++s
    ) {
        CUDA_CHECK(
            cudaFree(d_A[s])
        );

        CUDA_CHECK(
            cudaFree(d_B[s])
        );

        CUDA_CHECK(
            cudaFree(d_C[s])
        );

        CUDA_CHECK(
            cudaStreamDestroy(
                streams[s]
            )
        );
    }


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
