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


int main() {

    constexpr int NUM_SLOTS = 2;

    constexpr int NUM_REQUESTS = 8;

    constexpr int N = 1 << 20;

    constexpr int BLOCK_SIZE = 256;

    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(float);

    float* h_input[NUM_SLOTS]{};
    float* h_output[NUM_SLOTS]{};

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {

        CUDA_CHECK(cudaMallocHost(&h_input[slot], bytes));
        CUDA_CHECK(cudaMallocHost(&h_output[slot], bytes));
    }

    float* d_input[NUM_SLOTS]{};
    float* d_output[NUM_SLOTS]{};

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {

        CUDA_CHECK(cudaMalloc(&d_input[slot], bytes));
        CUDA_CHECK(cudaMalloc(&d_output[slot], bytes));
    }

    cudaStream_t streams[NUM_SLOTS];

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {
        
        CUDA_CHECK(cudaStreamCreate(&streams[slot]));
    }

    cudaEvent_t slot_done[NUM_SLOTS];

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {

        CUDA_CHECK(cudaEventCreateWithFlags(
                &slot_done[slot],
                cudaEventdisableTiming));
    }

    const int blocks =
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int request = 0; request < NUM_REQUESTS; ++request) {

        const int slot = request % NUM_SLOTS;
    

        if (request >= NUM_SLOTS) {

            CUDA_CHECK(cudaEventSynchronize(slot_done[slot]));
        }

        const float input_value = static_cast<float>(request);

        for (int idx = 0; idx < N; ++idx) {

            h_input[slot][idx] = input_value;
        }

        CUDA_CHECK(cudaMemcpyAsync(
                d_input[slot], 
                h_input[slot], 
                bytes, 
                cudaMemcpyHostToDevice, 
                streams[slot]));

        fake_inference_kernel<<<
            blocks,
            BLOCK_SIZE,
            0,
            streams[slot]
        >>>(
            d_input[slot],
            d_output[slot],
            N);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
                h_output[slot],
                d_output[slot],
                bytes,
                cudaMemcpyDeviceToHost,
                streams[slot]));

        CUDA_CHECK(cudaEventRecord(slot_done[slot], streams[slot]));
    }

    for (int slot = 0; slot < NUM_SLOTS; ++slot) {

        CUDA_CHECK(cudaEventSynchronize(slot_done[slot]));
    }
