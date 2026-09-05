__global__ void bias_relu_kernel(
    const float* input, const float* bias, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float value = input[idx] + bias[idx];

    output[idx] = fmaxf(value, 0.0F);
}
