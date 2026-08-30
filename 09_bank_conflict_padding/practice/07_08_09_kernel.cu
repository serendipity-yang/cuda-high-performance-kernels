







__global__ void transpose_naive(
    const float* input, float* output, int width, int height) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {

        output[x * height + y] = input[y * width + x];
    }
}

__global__ void transpose_tiled(
    const float* input, float* output, int width, int height) {

    constexpr int TILE_DIM = 32;

    __shared__ float tile[TILE_DIM][TILE_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * TILE_DIM + tx;
    const int y = blockIdx.y * TILE_DIM + ty;

    if (x < width && y < height) {

        tile[ty][tx] = input[y * width + x];
    }
    
    __syncthreads();

    const int out_x = blockIdx.y * TILE_DIM + tx;
    const int out_y = blockIdx.x * TILE_DIM + ty;

    if (out_x < height && out_y < width) {

        output[out_y * height + out_x] = tile[tx][ty];



