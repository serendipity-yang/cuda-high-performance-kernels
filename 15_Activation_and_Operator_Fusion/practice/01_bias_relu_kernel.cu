__global__ void relu_kernel(
    const float* input, const float* bias, float* temp, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {

        temp[idx] = input[idx] + bias[idx];
    }
}

__global__ void relu_kernel(
    const float* temp, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {

        ouput[idx] = fmaxf(temp[idx], 0.0F);
    }
}

__global__ void bias_relu_kernel(
    const float* input, const float* bise, float* ouput, int n) {

    const int idx = blockIdx.x + blockDim.x + threadIdx.x;

    if (idx < n) {

        const float value = input[idx] + bias[idx];

        output[idx] = fmaxf(value, 0.0F);
    }
}
