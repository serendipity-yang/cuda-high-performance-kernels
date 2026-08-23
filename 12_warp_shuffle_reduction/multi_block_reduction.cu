#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>


//================================================
// CUDA Error Check
//================================================

#define CUDA_CHECK(call)                                      \
do                                                            \
{                                                             \
    cudaError_t error = (call);                               \
                                                              \
    if(error != cudaSuccess)                                  \
    {                                                         \
        std::cerr                                             \
            << "CUDA Error: "                                 \
            << cudaGetErrorString(error)                       \
            << "\n";                                          \
                                                              \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
                                                              \
}while(0)



//================================================
// Warp Reduction
//================================================

__inline__ __device__
float warp_reduce_sum(float value)
{

    /*
        一个warp:

        lane0
        lane1
        ...
        lane31


        通过shuffle:

        32
        |
        16
        |
        8
        |
        4
        |
        2
        |
        1

    */


    unsigned int mask =
        0xffffffff;


    for(
        int offset = warpSize / 2;
        offset > 0;
        offset /= 2
    )
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




//================================================
// Block Reduction Kernel
//================================================


__global__
void reduce_kernel(
    const float* input,
    float* output,
    int n
)
{

    /*
    
    一个block:

        256 threads


    每个thread读取两个元素:


        thread0:

            input[0]
            input[256]


        thread1:

            input[1]
            input[257]



    */




    extern __shared__
    float shared[];




    int tid =
        threadIdx.x;



    int idx =
        blockIdx.x *
        blockDim.x *
        2
        +
        tid;



    //------------------------------------
    // Step 1
    // 每个线程加载两个元素
    //------------------------------------


    float sum = 0.0f;



    if(idx < n)
    {
        sum += input[idx];
    }



    if(
        idx + blockDim.x < n
    )
    {
        sum +=
            input[idx + blockDim.x];
    }





    //------------------------------------
    // Step 2
    // Warp Reduction
    //------------------------------------


    sum =
        warp_reduce_sum(sum);




    //------------------------------------
    // Step 3
    // 每个warp保存lane0结果
    //------------------------------------


    int lane =
        tid % warpSize;



    int warp_id =
        tid / warpSize;




    if(lane == 0)
    {

        shared[warp_id]
            =
            sum;

    }



    __syncthreads();





    //------------------------------------
    // Step 4
    // 第一个warp继续reduce
    //------------------------------------


    if(warp_id == 0)
    {


        int num_warps =
            blockDim.x / warpSize;



        if(lane < num_warps)
        {

            sum =
                shared[lane];

        }
        else
        {

            sum = 0.0f;

        }



        sum =
            warp_reduce_sum(sum);



        if(lane == 0)
        {

            output[blockIdx.x]
                =
                sum;

        }

    }

}





//================================================
// Main
//================================================


int main()
{


    constexpr int n =
        1 << 22;



    constexpr int BLOCK_SIZE =
        256;



    //--------------------------------
    // 计算block数量
    //--------------------------------


    int blocks =
        (n + BLOCK_SIZE * 2 - 1)
        /
        (BLOCK_SIZE * 2);



    std::cout
        << "n="
        << n
        << "\n";


    std::cout
        << "blocks="
        << blocks
        << "\n";





    //--------------------------------
    // Host data
    //--------------------------------


    std::vector<float>
        h_input(
            n,
            1.0f
        );





    //--------------------------------
    // Device memory
    //--------------------------------


    float* d_input =
        nullptr;


    float* d_partial =
        nullptr;



    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            n*sizeof(float)
        )
    );



    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            n*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );




    CUDA_CHECK(
        cudaMalloc(
            &d_partial,
            blocks*sizeof(float)
        )
    );






    //--------------------------------
    // Kernel 1
    //--------------------------------



    reduce_kernel
    <<<

        blocks,

        BLOCK_SIZE,

        (BLOCK_SIZE/32)
        *
        sizeof(float)

    >>>
    (

        d_input,

        d_partial,

        n

    );





    CUDA_CHECK(
        cudaDeviceSynchronize()
    );






    //--------------------------------
    // 第二阶段
    //--------------------------------


    int remaining =
        blocks;



    while(
        remaining > 1
    )
    {


        int new_blocks =
            (
                remaining
                +
                BLOCK_SIZE*2
                -
                1
            )
            /
            (
                BLOCK_SIZE*2
            );



        reduce_kernel
        <<<

            new_blocks,

            BLOCK_SIZE,

            (BLOCK_SIZE/32)
            *
            sizeof(float)

        >>>
        (

            d_partial,

            d_partial,

            remaining

        );



        remaining =
            new_blocks;


    }





    //--------------------------------
    // Copy result
    //--------------------------------


    float gpu_result;



    CUDA_CHECK(
        cudaMemcpy(
            &gpu_result,
            d_partial,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );






    //--------------------------------
    // CPU check
    //--------------------------------


    float cpu_result=0;



    for(float x:h_input)
    {

        cpu_result += x;

    }





    std::cout
        << "CPU="
        << cpu_result
        << "\n";


    std::cout
        << "GPU="
        << gpu_result
        << "\n";



    std::cout
        << "correct="
        << std::boolalpha
        << (
            fabs(
                cpu_result-gpu_result
            )
            <
            1e-5
        )
        << "\n";





    cudaFree(d_input);

    cudaFree(d_partial);


    return 0;

}
