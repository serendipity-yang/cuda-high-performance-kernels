//ReLU
__global__ void relu_kernel(
    const float* input, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float x = input[idx];

    ouput[idx] = fmax(input[idx], 0.0F);
}

//Sigmoid
__global__ void sigmoid_kernel(
    const float* input, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float x = input[idx];

    const float sigmoid_value = 1.0F / (1.0F + expf(-x));

    output[idx] = sigmoid_value;
}

//Tanh
__global__ void tanh_kernel(
    const float* input, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float x = input[idx];

    ouput[idx] = tanhf(input[idx]);
}

//GELU
__global__ void gelu_kernel(
    const float* input, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float x = input[idx];

    const float x3 = x * x * x;

    const float inner = x + 0.044715 * x3;

    const float tanh_value = tanhf(0.7978845608F * inner);

    const float gelu_value = 0.5F * x * (1.0F + tanh_value);

    output[idx] = gelu_value;
}

//SiLU
__global__ void silu_kernel(
    const float* input, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float x = input[idx];

    const float sigmiod_value = 1.0F / (1.0F + exp(-x));

    const float silu_value = x * sigmiod_value;

    ouput[idx] = silu_value;
}




