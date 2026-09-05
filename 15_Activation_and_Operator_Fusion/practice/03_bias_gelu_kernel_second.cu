__global__ void bias_gelu_kernel(
    const float* input, const float* bias, float* output, int n) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) {

        return;
    }

    const float x = input[idx];

    const float x3 = x * x * x;

    const float inner = x + 0.044715 * x3;

    const float tanh_value = tanhf(0.7978845608 * inner);

    const float gelu_value = 0.5F * x * (1 + tanh_value);

    output[idx] = gelu_value;
}
