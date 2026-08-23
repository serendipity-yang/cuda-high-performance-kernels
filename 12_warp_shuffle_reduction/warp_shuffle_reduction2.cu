#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>


#define CUDA_CHECK(call)                         \
do {                                             \
    cudaError_t error = (call);                  \
    if(error != cudaSuccess){                    \
        std::cerr                              \
            << "CUDA Error: "                   \
            << cudaGetErrorString(error)         \
            << "\n";                            \
        std::exit(EXIT_FAILURE);                 \
    }                                            \
}while(0)


//----------------------------------
// Warp Reduce
//----------------------------------

__inline__ __device__
float warp_reduce_sum(float value)
{
    const unsigned int mask =
        0xffffffffu;


    for(int offset = warpSize / 2;
        offset > 0;
        offset /= 2)
    {

        value +=
            __shfl_down_sync(
                mask,
                value,
                offset
            );
    }


    return value;
}



//----------------------------------
// Block Reduction
//----------------------------------

__global__
void warp_shuffle_reduction(
    const float* input,
    float* block_sums,
    int n
)
{

    extern __shared__ float shared[];


    int tid =
        threadIdx.x;


    int idx =
        blockIdx.x *
        blockDim.x *
        2
        +
        tid;



    //--------------------------------
    // 每个线程读取两个元素
    //--------------------------------

    float sum = 0.0f;


    if(idx < n)
    {
        sum += input[idx];
    }


    if(idx + blockDim.x < n)
    {
        sum += input[idx + blockDim.x];
    }


    //--------------------------------
    // Warp 内 reduction
    //--------------------------------

    sum =
        warp_reduce_sum(sum);



    //--------------------------------
    // 每个warp的lane0保存结果
    //--------------------------------

    int lane =
        tid % warpSize;


    int warp_id =
        tid / warpSize;



    if(lane == 0)
    {
        shared[warp_id] = sum;
    }


    __syncthreads();



    //--------------------------------
    // 第一个warp继续reduce
    //--------------------------------

    float block_sum = 0.0f;


    if(warp_id == 0)
    {

        if(tid < blockDim.x / warpSize)
        {
            block_sum =
                shared[lane];
        }


        block_sum =
            warp_reduce_sum(block_sum);


        if(lane == 0)
        {
            block_sums[blockIdx.x]
                =
                block_sum;
        }
    }

}



int main()
{

    int n =
        1 << 20;


    size_t bytes =
        n * sizeof(float);



    std::vector<float> h_input(n);


    for(int i=0;i<n;i++)
    {
        h_input[i]=1.0f;
    }



    float* d_input;

    float* d_block;



    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            bytes
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );



    int BLOCK_SIZE = 256;


    int num_blocks =
        (n + BLOCK_SIZE*2 -1)
        /
        (BLOCK_SIZE*2);



    CUDA_CHECK(
        cudaMalloc(
            &d_block,
            num_blocks*sizeof(float)
        )
    );



    warp_shuffle_reduction
    <<<
        num_blocks,
        BLOCK_SIZE,
        (BLOCK_SIZE/32)*sizeof(float)
    >>>
    (
        d_input,
        d_block,
        n
    );



    CUDA_CHECK(
        cudaDeviceSynchronize()
    );



    std::vector<float>
        h_block(num_blocks);



    CUDA_CHECK(
        cudaMemcpy(
            h_block.data(),
            d_block,
            num_blocks*sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );



    float result = 0;


    for(float x:h_block)
    {
        result += x;
    }



    std::cout
        << "GPU sum = "
        << result
        << std::endl;



    CUDA_CHECK(
        cudaFree(d_input)
    );


    CUDA_CHECK(
        cudaFree(d_block)
    );

}

