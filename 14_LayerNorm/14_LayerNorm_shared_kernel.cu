__global__ void layernorm_kernel(
    const float* intput, const float* gamma, const float* beta,
    float* output, int rows, int cols, float epsilon) {

    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    const int row = blockIdx.x;

    if (row >= rows) {

        return;
    }

    const int col = tid;

    const int idx = row * cols + row;

    float value = 0.0F;

    if (col < cols) {

        value = input[idx];
    }

    shared[tid] = value;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {

            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    const float mean = shared[0] / static_cast<float>(cols);

    float diff = 0.0F;

    float squared = 0.0F;

    if (col < cols) {

        diff = input[idx] - mean;

        squared = diff * diff;
    }

    shared[tid] = squared;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {

            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    const float variance = shared[0] / cols;

    const float intv_std = rsqrtf(variance + epsion);

    if (col < cols) {

        const float normaliazed = diff * inv_std;

        output[idx] = normaliazed * gamma[col] + beta[col];
    }
}



