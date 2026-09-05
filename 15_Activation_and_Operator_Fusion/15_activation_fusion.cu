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


constexpr int BLOCK_SIZE =
    256;


//============================================================
// Kernel 1
// Bias
//
// temp[idx] = input[idx] + bias[idx]
//
// 这是未融合版本的第一个 Kernel
//============================================================

__global__
void bias_kernel(
    const float* input,
    const float* bias,
    float* temp,
    int n
)
{
    const int idx =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    if (idx >= n)
    {
        return;
    }


    temp[idx] =
        input[idx]
        +
        bias[idx];
}


//============================================================
// Kernel 2
// ReLU
//
// output[idx] = max(temp[idx], 0)
//
// 这是未融合版本的第二个 Kernel
//============================================================

__global__
void relu_kernel(
    const float* temp,
    float* output,
    int n
)
{
    const int idx =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    if (idx >= n)
    {
        return;
    }


    output[idx] =
        fmaxf(
            temp[idx],
            0.0F
        );
}


//============================================================
// Kernel 3
// Fused Bias + ReLU
//
// output[idx]
//     = ReLU(input[idx] + bias[idx])
//
// 不再需要 temp Global Memory
//============================================================

__global__
void bias_relu_fused_kernel(
    const float* input,
    const float* bias,
    float* output,
    int n
)
{
    const int idx =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    if (idx >= n)
    {
        return;
    }


    const float value =
        input[idx]
        +
        bias[idx];


    output[idx] =
        fmaxf(
            value,
            0.0F
        );
}


//============================================================
// Kernel 4
// Fused Bias + GELU
//
// output[idx]
//     = GELU(input[idx] + bias[idx])
//
// 使用 GELU tanh 近似公式
//============================================================

__global__
void bias_gelu_fused_kernel(
    const float* input,
    const float* bias,
    float* output,
    int n
)
{
    const int idx =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    if (idx >= n)
    {
        return;
    }


    //========================================================
    // Bias
    //========================================================

    const float x =
        input[idx]
        +
        bias[idx];


    //========================================================
    // GELU
    //
    // GELU(x)
    // ≈ 0.5 * x *
    //   (1 + tanh(
    //       sqrt(2/pi) *
    //       (x + 0.044715*x^3)
    //   ))
    //========================================================

    const float x3 =
        x
        *
        x
        *
        x;


    const float inner =
        x
        +
        0.044715F
        *
        x3;


    const float tanh_value =
        tanhf(
            0.7978845608F
            *
            inner
        );


    const float gelu_value =
        0.5F
        *
        x
        *
        (
            1.0F
            +
            tanh_value
        );


    output[idx] =
        gelu_value;
}


//============================================================
// CPU Reference
// Bias + ReLU
//============================================================

void bias_relu_cpu(
    const std::vector<float>& input,
    const std::vector<float>& bias,
    std::vector<float>& output
)
{
    const int n =
        static_cast<int>(
            input.size()
        );


    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        const float value =
            input[idx]
            +
            bias[idx];


        output[idx] =
            std::fmax(
                value,
                0.0F
            );
    }
}


//============================================================
// CPU GELU
//============================================================

float gelu_cpu(
    float x
)
{
    const float x3 =
        x
        *
        x
        *
        x;


    const float inner =
        x
        +
        0.044715F
        *
        x3;


    const float tanh_value =
        std::tanh(
            0.7978845608F
            *
            inner
        );


    return
        0.5F
        *
        x
        *
        (
            1.0F
            +
            tanh_value
        );
}


//============================================================
// CPU Reference
// Bias + GELU
//============================================================

void bias_gelu_cpu(
    const std::vector<float>& input,
    const std::vector<float>& bias,
    std::vector<float>& output
)
{
    const int n =
        static_cast<int>(
            input.size()
        );


    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        const float x =
            input[idx]
            +
            bias[idx];


        output[idx] =
            gelu_cpu(x);
    }
}


//============================================================
// main
//============================================================

int main()
{
    const int n =
        1024;


    const std::size_t bytes =
        static_cast<std::size_t>(n)
        *
        sizeof(float);


    //========================================================
    // Host Memory
    //========================================================

    std::vector<float> h_input(n);

    std::vector<float> h_bias(n);


    std::vector<float>
        h_cpu_relu(n);

    std::vector<float>
        h_gpu_unfused_relu(n);

    std::vector<float>
        h_gpu_fused_relu(n);


    std::vector<float>
        h_cpu_gelu(n);

    std::vector<float>
        h_gpu_fused_gelu(n);


    //========================================================
    // Initialize Input
    //========================================================

    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        h_input[idx] =
            static_cast<float>(
                idx % 11
            )
            -
            5.0F;


        h_bias[idx] =
            1.0F;
    }


    //========================================================
    // CPU Reference
    //========================================================

    bias_relu_cpu(
        h_input,
        h_bias,
        h_cpu_relu
    );


    bias_gelu_cpu(
        h_input,
        h_bias,
        h_cpu_gelu
    );


    //========================================================
    // Device Pointers
    //========================================================

    float* d_input =
        nullptr;

    float* d_bias =
        nullptr;

    float* d_temp =
        nullptr;

    float* d_unfused_relu =
        nullptr;

    float* d_fused_relu =
        nullptr;

    float* d_fused_gelu =
        nullptr;


    //========================================================
    // cudaMalloc
    //========================================================

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_bias,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_temp,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_unfused_relu,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_fused_relu,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_fused_gelu,
            bytes
        )
    );


    //========================================================
    // Host -> Device
    //========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_bias,
            h_bias.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    //========================================================
    // Grid
    //========================================================

    const int num_blocks =
        (
            n
            +
            BLOCK_SIZE
            -
            1
        )
        /
        BLOCK_SIZE;


    //========================================================
    // Experiment 1
    // Unfused Bias + ReLU
    //
    // Kernel 1:
    // input + bias -> temp
    //
    // Kernel 2:
    // temp -> ReLU -> output
    //========================================================

    bias_kernel<<<
        num_blocks,
        BLOCK_SIZE
    >>>(
        d_input,
        d_bias,
        d_temp,
        n
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    relu_kernel<<<
        num_blocks,
        BLOCK_SIZE
    >>>(
        d_temp,
        d_unfused_relu,
        n
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //========================================================
    // Experiment 2
    // Fused Bias + ReLU
    //========================================================

    bias_relu_fused_kernel<<<
        num_blocks,
        BLOCK_SIZE
    >>>(
        d_input,
        d_bias,
        d_fused_relu,
        n
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //========================================================
    // Experiment 3
    // Fused Bias + GELU
    //========================================================

    bias_gelu_fused_kernel<<<
        num_blocks,
        BLOCK_SIZE
    >>>(
        d_input,
        d_bias,
        d_fused_gelu,
        n
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    //========================================================
    // Device -> Host
    //========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_gpu_unfused_relu.data(),
            d_unfused_relu,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_gpu_fused_relu.data(),
            d_fused_relu,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_gpu_fused_gelu.data(),
            d_fused_gelu,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    //========================================================
    // Correctness
    //========================================================

    bool unfused_relu_correct =
        true;


    bool fused_relu_correct =
        true;


    bool fused_gelu_correct =
        true;


    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        if (
            std::fabs(
                h_cpu_relu[idx]
                -
                h_gpu_unfused_relu[idx]
            )
            >
            1e-5F
        )
        {
            unfused_relu_correct =
                false;

            break;
        }
    }


    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        if (
            std::fabs(
                h_cpu_relu[idx]
                -
                h_gpu_fused_relu[idx]
            )
            >
            1e-5F
        )
        {
            fused_relu_correct =
                false;

            break;
        }
    }


    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        if (
            std::fabs(
                h_cpu_gelu[idx]
                -
                h_gpu_fused_gelu[idx]
            )
            >
            1e-5F
        )
        {
            fused_gelu_correct =
                false;

            break;
        }
    }


    //========================================================
    // Print
    //========================================================

    std::cout
        << std::boolalpha;


    std::cout
        << "Unfused Bias + ReLU correct = "
        << unfused_relu_correct
        << '\n';


    std::cout
        << "Fused Bias + ReLU correct   = "
        << fused_relu_correct
        << '\n';


    std::cout
        << "Fused Bias + GELU correct   = "
        << fused_gelu_correct
        << '\n';


    std::cout
        << "\nFirst 8 Bias + ReLU outputs:\n";


    for (
        int idx = 0;
        idx < 8;
        ++idx
    )
    {
        std::cout
            << h_gpu_fused_relu[idx]
            << ' ';
    }


    std::cout
        << '\n';


    //========================================================
    // Free
    //========================================================

    CUDA_CHECK(
        cudaFree(
            d_input
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_bias
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_temp
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_unfused_relu
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_fused_relu
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_fused_gelu
        )
    );


    return 0;
}
