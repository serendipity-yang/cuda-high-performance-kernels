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
void fake_inference_kernel(
    const float* input,
    float* output,
    int n
)
{
    const int idx =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;

    if (idx < n) {
        // 模拟一个非常简单的“模型”
        output[idx]
            =
            input[idx] * 2.0F
            +
            1.0F;
    }
}


int main()
{
    constexpr int NUM_SLOTS =
        2;

    constexpr int NUM_REQUESTS =
        8;

    constexpr int N =
        1 << 20;

    constexpr int BLOCK_SIZE =
        256;


    const std::size_t bytes =
        static_cast<std::size_t>(N)
        *
        sizeof(float);


    //==================================================
    // Two pinned host slots
    //==================================================

    float* h_input[NUM_SLOTS]{};
    float* h_output[NUM_SLOTS]{};


    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        CUDA_CHECK(
            cudaMallocHost(
                &h_input[slot],
                bytes
            )
        );

        CUDA_CHECK(
            cudaMallocHost(
                &h_output[slot],
                bytes
            )
        );
    }


    //==================================================
    // Two device slots
    //==================================================

    float* d_input[NUM_SLOTS]{};
    float* d_output[NUM_SLOTS]{};


    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        CUDA_CHECK(
            cudaMalloc(
                &d_input[slot],
                bytes
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d_output[slot],
                bytes
            )
        );
    }


    //==================================================
    // Two Streams
    //==================================================

    cudaStream_t streams[NUM_SLOTS];


    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        CUDA_CHECK(
            cudaStreamCreate(
                &streams[slot]
            )
        );
    }


    //==================================================
    // One completion event per slot
    //
    // Event tells us:
    // previous request using this slot is completely done.
    //==================================================

    cudaEvent_t slot_done[NUM_SLOTS];


    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        CUDA_CHECK(
            cudaEventCreateWithFlags(
                &slot_done[slot],
                cudaEventDisableTiming
            )
        );
    }


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
    // Submit Requests
    //==================================================

    for (
        int request = 0;
        request < NUM_REQUESTS;
        ++request
    ) {
        const int slot =
            request
            %
            NUM_SLOTS;


        //================================================
        // Request >= 2 means this slot has already been
        // used before.
        //
        // Before CPU overwrites h_input[slot],
        // ensure previous asynchronous operations
        // using this slot are complete.
        //================================================

        if (
            request
            >=
            NUM_SLOTS
        ) {
            CUDA_CHECK(
                cudaEventSynchronize(
                    slot_done[slot]
                )
            );
        }


        //================================================
        // CPU prepares current request.
        //
        // Different request uses different value,
        // so we can verify slot reuse is correct.
        //================================================

        const float input_value =
            static_cast<float>(
                request
            );


        for (
            int idx = 0;
            idx < N;
            ++idx
        ) {
            h_input[slot][idx]
                =
                input_value;
        }


        std::cout
            << "Submit request "
            << request
            << " -> slot "
            << slot
            << '\n';


        //================================================
        // 1. H2D
        //================================================

        CUDA_CHECK(
            cudaMemcpyAsync(
                d_input[slot],
                h_input[slot],
                bytes,
                cudaMemcpyHostToDevice,
                streams[slot]
            )
        );


        //================================================
        // 2. Inference
        //
        // Equivalent conceptually to:
        //
        // context->enqueueV3(streams[slot]);
        //================================================

        fake_inference_kernel<<<
            blocks,
            BLOCK_SIZE,
            0,
            streams[slot]
        >>>(
            d_input[slot],
            d_output[slot],
            N
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        //================================================
        // 3. D2H
        //================================================

        CUDA_CHECK(
            cudaMemcpyAsync(
                h_output[slot],
                d_output[slot],
                bytes,
                cudaMemcpyDeviceToHost,
                streams[slot]
            )
        );


        //================================================
        // 4. Record:
        //
        // this slot is safe to reuse only after
        // H2D + Kernel + D2H are all complete.
        //================================================

        CUDA_CHECK(
            cudaEventRecord(
                slot_done[slot],
                streams[slot]
            )
        );
    }


    //==================================================
    // Wait final two requests
    //==================================================

    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        CUDA_CHECK(
            cudaEventSynchronize(
                slot_done[slot]
            )
        );
    }


    //==================================================
    // Final slot contents correspond to:
    //
    // slot0 -> request6
    // slot1 -> request7
    //==================================================

    bool correct =
        true;


    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        const int final_request =
            NUM_REQUESTS
            -
            NUM_SLOTS
            +
            slot;


        const float expected =
            static_cast<float>(
                final_request
            )
            *
            2.0F
            +
            1.0F;


        for (
            int idx = 0;
            idx < N;
            ++idx
        ) {
            if (
                std::fabs(
                    h_output[slot][idx]
                    -
                    expected
                )
                >
                1e-5F
            ) {
                correct =
                    false;

                std::cerr
                    << "slot = "
                    << slot
                    << ", idx = "
                    << idx
                    << ", expected = "
                    << expected
                    << ", actual = "
                    << h_output[slot][idx]
                    << '\n';

                break;
            }
        }


        if (!correct) {
            break;
        }
    }


    std::cout
        << std::boolalpha
        << "correct = "
        << correct
        << '\n';


    //==================================================
    // Cleanup
    //==================================================

    for (
        int slot = 0;
        slot < NUM_SLOTS;
        ++slot
    ) {
        CUDA_CHECK(
            cudaEventDestroy(
                slot_done[slot]
            )
        );

        CUDA_CHECK(
            cudaStreamDestroy(
                streams[slot]
            )
        );

        CUDA_CHECK(
            cudaFree(
                d_input[slot]
            )
        );

        CUDA_CHECK(
            cudaFree(
                d_output[slot]
            )
        );

        CUDA_CHECK(
            cudaFreeHost(
                h_input[slot]
            )
        );

        CUDA_CHECK(
            cudaFreeHost(
                h_output[slot]
            )
        );
    }


    return 0;
}
