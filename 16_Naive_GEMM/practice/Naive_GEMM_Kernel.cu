__global__ void naive_gemm_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) {

    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {

        return;
    }

    float sum = 0.0F;

    for (int k = 0; k < K; ++k) {

        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}
