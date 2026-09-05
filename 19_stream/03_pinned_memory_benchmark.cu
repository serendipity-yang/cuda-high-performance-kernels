#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
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


double calculate_bandwidth_gbps(
    std::size_t bytes,
    int repeat,
    double milliseconds
)
{
    const double total_bytes =
        static_cast<double>(bytes)
        *
        repeat;

    const double seconds =
        milliseconds
        /
        1000.0;

    return
        total_bytes
        /
        seconds
        /
        1e9;
}


int main()
{
    // 2^24 float
    // = 16,777,216 floats
    // ≈ 64 MiB
    constexpr std::size_t N =
        1ULL << 24;

    constexpr int WARMUP =
        3;

    constexpr int REPEAT =
        20;

    const std::size_t bytes =
        N
        *
        sizeof(float);


    std::cout
        << "Transfer size per copy = "
        << static_cast<double>(bytes) / 1e6
        << " MB\n";


    //==================================================
    // Pageable Host Memory
    //==================================================

    std::vector<float> h_pageable(
        N,
        1.0F
    );


    //==================================================
    // Pinned Host Memory
    //==================================================

    float* h_pinned =
        nullptr;

    CUDA_CHECK(
        cudaMallocHost(
            &h_pinned,
            bytes
        )
    );

    for (
        std::size_t idx = 0;
        idx < N;
        ++idx
    ) {
        h_pinned[idx] =
            1.0F;
    }


    //==================================================
    // Device Memory
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
    // Stream
    //==================================================

    cudaStream_t stream;

    CUDA_CHECK(
        cudaStreamCreate(
            &stream
        )
    );


    //==================================================
    // Warmup
    //==================================================

    for (
        int i = 0;
        i < WARMUP;
        ++i
    ) {
        CUDA_CHECK(
            cudaMemcpy(
                d_data,
                h_pageable.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );
    }


    for (
        int i = 0;
        i < WARMUP;
        ++i
    ) {
        CUDA_CHECK(
            cudaMemcpyAsync(
                d_data,
                h_pinned,
                bytes,
                cudaMemcpyHostToDevice,
                stream
            )
        );
    }

    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );


    //==================================================
    // Pageable H2D
    //==================================================

    auto pageable_h2d_start =
        std::chrono::steady_clock::now();


    for (
        int i = 0;
        i < REPEAT;
        ++i
    ) {
        CUDA_CHECK(
            cudaMemcpy(
                d_data,
                h_pageable.data(),
                bytes,
                cudaMemcpyHostToDevice
            )
        );
    }


    auto pageable_h2d_stop =
        std::chrono::steady_clock::now();


    const double pageable_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            pageable_h2d_stop
            -
            pageable_h2d_start
        ).count();


    //==================================================
    // Pinned H2D
    //==================================================

    auto pinned_h2d_start =
        std::chrono::steady_clock::now();


    for (
        int i = 0;
        i < REPEAT;
        ++i
    ) {
        CUDA_CHECK(
            cudaMemcpyAsync(
                d_data,
                h_pinned,
                bytes,
                cudaMemcpyHostToDevice,
                stream
            )
        );
    }


    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );


    auto pinned_h2d_stop =
        std::chrono::steady_clock::now();


    const double pinned_h2d_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            pinned_h2d_stop
            -
            pinned_h2d_start
        ).count();


    //==================================================
    // Pageable D2H
    //==================================================

    auto pageable_d2h_start =
        std::chrono::steady_clock::now();


    for (
        int i = 0;
        i < REPEAT;
        ++i
    ) {
        CUDA_CHECK(
            cudaMemcpy(
                h_pageable.data(),
                d_data,
                bytes,
                cudaMemcpyDeviceToHost
            )
        );
    }


    auto pageable_d2h_stop =
        std::chrono::steady_clock::now();


    const double pageable_d2h_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            pageable_d2h_stop
            -
            pageable_d2h_start
        ).count();


    //==================================================
    // Pinned D2H
    //==================================================

    auto pinned_d2h_start =
        std::chrono::steady_clock::now();


    for (
        int i = 0;
        i < REPEAT;
        ++i
    ) {
        CUDA_CHECK(
            cudaMemcpyAsync(
                h_pinned,
                d_data,
                bytes,
                cudaMemcpyDeviceToHost,
                stream
            )
        );
    }


    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );


    auto pinned_d2h_stop =
        std::chrono::steady_clock::now();


    const double pinned_d2h_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            pinned_d2h_stop
            -
            pinned_d2h_start
        ).count();


    //==================================================
    // Print
    //==================================================

    std::cout
        << std::fixed
        << std::setprecision(3);


    std::cout
        << "\n===== H2D =====\n";

    std::cout
        << "Pageable average = "
        << pageable_h2d_ms / REPEAT
        << " ms, "
        << calculate_bandwidth_gbps(
            bytes,
            REPEAT,
            pageable_h2d_ms
        )
        << " GB/s\n";

    std::cout
        << "Pinned average   = "
        << pinned_h2d_ms / REPEAT
        << " ms, "
        << calculate_bandwidth_gbps(
            bytes,
            REPEAT,
            pinned_h2d_ms
        )
        << " GB/s\n";


    std::cout
        << "\n===== D2H =====\n";

    std::cout
        << "Pageable average = "
        << pageable_d2h_ms / REPEAT
        << " ms, "
        << calculate_bandwidth_gbps(
            bytes,
            REPEAT,
            pageable_d2h_ms
        )
        << " GB/s\n";

    std::cout
        << "Pinned average   = "
        << pinned_d2h_ms / REPEAT
        << " ms, "
        << calculate_bandwidth_gbps(
            bytes,
            REPEAT,
            pinned_d2h_ms
        )
        << " GB/s\n";


    CUDA_CHECK(
        cudaStreamDestroy(
            stream
        )
    );

    CUDA_CHECK(
        cudaFree(d_data)
    );

    CUDA_CHECK(
        cudaFreeHost(h_pinned)
    );


    return 0;
}
