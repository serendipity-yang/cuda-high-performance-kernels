








__global__ void reduce_naive(
    const float* input, float* output, int n) {

    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (idx < n) {

        sum += input[idx];
    }

    if (idx + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }

    shared[tid] = sum;

    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        if (tid % (2 * stride) == 0) {

            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {

        output[blockIdx.x] = shared[0];
    }
}
















__global__ void reduce_shared(
    const float* input, float* output, int n) {

    extern __shared__ float shared[];

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (tid < n) {

        sum += input[idx];
    }

    if (tid + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }
    
    shared[tid] = sum;

    __syncthreads();

    for (int stride = blockDim / 2; stride > 0; stride << 1) {

        if (tid < stride) {

            shared[tid] += shared[tid + stride];
        }
    }

    __syncthreads();

    if (tid == 0) {

        ouput[blockDim.x] = shared[0];
    }
}





__inline__ __device__ float warp_reduce_sum (float value) {

    const unsigned int mask = 0xffffffffu;

    for (int offset = warpSize / 2; offset > 0; offset >> 1) {

        value += 
            __shfl_down_sync(mask, value, offset);
    }

    return value;
}

__global__ void reduce_warp_shuffle(
    const float* input, float* output, int n) {

    __shared__ float warp_sums[32];

    const int tid = threadIdx.x;

    const int idx = blockIdx.x * blockDim.x * 2 + tid;

    float sum = 0.0F;

    if (tid < n) {
        
        sum += input[idx];
    }

    if (tid + blockDim.x < n) {

        sum += input[idx + blockDim.x];
    }

    sum = warp_reduce_sum(sum);
    
    const int lane = tid % warpSize;

    const int warp_id = tid / warpSize;

    if (lane == 0) {

        warp_sum[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0) {

        const int num_warps = 

            (blockDim.x + warpSize - 1) / warpSize;

            sum = 
                (lane < num_warps) ? warp_sums[lane] : 0.0F;

            sum = warp_reduce_sum(sum);

            if (lane == 0) {

                output[blockIdx.x] = sum;
            }
    }
}

