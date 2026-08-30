__global__ void softmax_kernel(
    const float* input,float* output, int rows, int cols) {

    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    const int row = blockIdx.x;

    const int col = tid;

    const int idx = row * cols + col;

    float value = -FLT_MAX;

    if (col < cols) {

        value = input[idx];
    }

    shared[tid] = value;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        
        if (tid < stride) {

            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }

        __syncthreads();
    }

    const float row_max = shared[0];

    float exp_value = 0.0F;

    if (col < cols) {

        exp_value = expf(input[idx] - row_max);
    }

    if (col < cols) {

        output[idx] = exp_value;
    }

    shared[tid] = exp_value;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {

            shared[tid] += shared[tid + stride];
    }

    __syncthreads();

    const float exp_sum = shared[0];
    
    if (col < cols) {

        output[idx] = output[idx] / exp_sum;
    }
}
