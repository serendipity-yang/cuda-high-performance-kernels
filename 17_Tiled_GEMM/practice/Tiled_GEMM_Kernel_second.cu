constexpr int TILE_SIZE = 2;

__global__ void tiled_gemm_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) {

    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];

    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;

    const int ty = threadIdx.y;

    const int row = blockIdx.y * TILE_SIZE + ty;

    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0F;

    const int num_phases = 

        (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int phase = 0; phase < num_phases; ++phase) {

        const int a_col = phase * TILE_SIZE + tx;

        const int b_row = phase * TILE_SIZE + ty;

        if (row < M && a_col < K) {

            shared_A[ty][tx] = A[row * K + a_col];
        } else {

            shared_A[ty][tx] = 0.0F;
        }

        if (b_row < K && col < N) {

            shared_B[ty][tx] = B[b_row * N + col];
        } else {

            shared_B[ty][tx] = 0.0F;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {

            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();

        if (row < M && col < N) {

            C[row * N + col] = sum;
        }
    }
}
