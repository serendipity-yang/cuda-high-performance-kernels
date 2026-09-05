#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>


//======================================================
// CUDA CHECK
//======================================================

#define CUDA_CHECK(call)                                      \
do                                                            \
{                                                             \
    cudaError_t error = (call);                               \
                                                              \
    if(error != cudaSuccess)                                  \
    {                                                         \
        std::cerr                                             \
            << "CUDA Error: "                                 \
            << cudaGetErrorString(error)                      \
            << "\n";                                          \
                                                              \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
                                                              \
}while(0)



//======================================================
// CUBLAS CHECK
//======================================================

#define CUBLAS_CHECK(call)                                    \
do                                                            \
{                                                             \
    cublasStatus_t status = (call);                           \
                                                              \
    if(status != CUBLAS_STATUS_SUCCESS)                       \
    {                                                         \
        std::cerr                                             \
            << "cuBLAS Error\n";                              \
                                                              \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
                                                              \
}while(0)





//======================================================
// Parameters
//======================================================


constexpr int M = 1024;

constexpr int N = 1024;

constexpr int K = 1024;


constexpr int TILE_SIZE = 16;


constexpr int WARMUP = 10;

constexpr int REPEAT = 100;





//======================================================
// Naive GEMM
//
// C = A * B
//
// A: M*K
// B: K*N
// C: M*N
//======================================================


__global__
void naive_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{

    const int row =
        blockIdx.y * blockDim.y
        +
        threadIdx.y;


    const int col =
        blockIdx.x * blockDim.x
        +
        threadIdx.x;



    if(row < M && col < N)
    {

        float sum = 0.0F;



        for(int k = 0;
            k < K;
            ++k)
        {

            sum +=
                A[row * K + k]
                *
                B[k * N + col];

        }



        C[row * N + col]
            =
            sum;

    }

}





//======================================================
// Tiled GEMM
//
// Shared Memory Optimization
//======================================================


__global__
void tiled_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K
)
{

    __shared__
    float shared_A[TILE_SIZE][TILE_SIZE];


    __shared__
    float shared_B[TILE_SIZE][TILE_SIZE];



    const int tx =
        threadIdx.x;


    const int ty =
        threadIdx.y;



    const int row =
        blockIdx.y * TILE_SIZE
        +
        ty;


    const int col =
        blockIdx.x * TILE_SIZE
        +
        tx;



    float sum = 0.0F;



    const int phases =
        (K + TILE_SIZE - 1)
        /
        TILE_SIZE;



    for(int phase = 0;
        phase < phases;
        ++phase)
    {


        const int a_col =
            phase * TILE_SIZE
            +
            tx;


        const int b_row =
            phase * TILE_SIZE
            +
            ty;



        //---------------------------------
        // Load A Tile
        //---------------------------------

        if(
            row < M &&
            a_col < K
        )
        {

            shared_A[ty][tx]
                =
                A[row*K+a_col];

        }
        else
        {

            shared_A[ty][tx]
                =
                0.0F;

        }



        //---------------------------------
        // Load B Tile
        //---------------------------------

        if(
            b_row < K &&
            col < N
        )
        {

            shared_B[ty][tx]
                =
                B[b_row*N+col];

        }
        else
        {

            shared_B[ty][tx]
                =
                0.0F;

        }



        __syncthreads();




        //---------------------------------
        // Compute Tile
        //---------------------------------

        for(int k = 0;
            k < TILE_SIZE;
            ++k)
        {

            sum +=
                shared_A[ty][k]
                *
                shared_B[k][tx];

        }



        __syncthreads();

    }



    if(
        row < M &&
        col < N
    )
    {

        C[row*N+col]
            =
            sum;

    }

}







//======================================================
// cuBLAS GEMM
//
// Row Major trick:
//
// C = A*B
//
// becomes:
//
// C^T = B^T*A^T
//======================================================


void cublas_gemm(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    float* d_C
)
{

    float alpha = 1.0F;

    float beta = 0.0F;



    CUBLAS_CHECK(
        cublasSgemm(

            handle,


            CUBLAS_OP_N,

            CUBLAS_OP_N,


            N,

            M,

            K,


            &alpha,


            d_B,

            N,


            d_A,

            K,


            &beta,


            d_C,

            N

        )
    );

}







//======================================================
// GFLOPS
//======================================================


double calculate_gflops(
    float milliseconds
)
{

    double operations =
        2.0
        *
        M
        *
        N
        *
        K;



    double seconds =
        milliseconds
        /
        1000.0;



    return
        operations
        /
        seconds
        /
        1e9;

}







//======================================================
// Benchmark Naive
//======================================================


float benchmark_naive(
    float* d_A,
    float* d_B,
    float* d_C,
    dim3 grid,
    dim3 block
)
{

    for(int i=0;
        i<WARMUP;
        ++i)
    {

        naive_gemm_kernel<<<grid,block>>>
        (
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );

    }



    CUDA_CHECK(
        cudaDeviceSynchronize()
    );



    cudaEvent_t start;

    cudaEvent_t stop;



    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);



    for(int i=0;
        i<REPEAT;
        ++i)
    {

        naive_gemm_kernel<<<grid,block>>>
        (
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );

    }



    cudaEventRecord(stop);



    cudaEventSynchronize(stop);



    float ms = 0.0F;



    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );



    cudaEventDestroy(start);

    cudaEventDestroy(stop);



    return ms / REPEAT;

}









//======================================================
// Benchmark Tiled
//======================================================


float benchmark_tiled(
    float* d_A,
    float* d_B,
    float* d_C,
    dim3 grid,
    dim3 block
)
{

    for(int i=0;i<WARMUP;i++)
    {

        tiled_gemm_kernel<<<grid,block>>>
        (
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );

    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    cudaEvent_t start;

    cudaEvent_t stop;


    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);



    for(int i=0;i<REPEAT;i++)
    {

        tiled_gemm_kernel<<<grid,block>>>
        (
            d_A,
            d_B,
            d_C,
            M,
            N,
            K
        );

    }



    cudaEventRecord(stop);


    cudaEventSynchronize(stop);



    float ms = 0.0F;


    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );



    cudaEventDestroy(start);

    cudaEventDestroy(stop);



    return ms / REPEAT;

}









//======================================================
// Benchmark cuBLAS
//======================================================


float benchmark_cublas(
    cublasHandle_t handle,
    float* d_A,
    float* d_B,
    float* d_C
)
{


    for(int i=0;i<WARMUP;i++)
    {

        cublas_gemm(
            handle,
            d_A,
            d_B,
            d_C
        );

    }



    CUDA_CHECK(
        cudaDeviceSynchronize()
    );



    cudaEvent_t start;

    cudaEvent_t stop;


    cudaEventCreate(&start);

    cudaEventCreate(&stop);



    cudaEventRecord(start);



    for(int i=0;i<REPEAT;i++)
    {

        cublas_gemm(
            handle,
            d_A,
            d_B,
            d_C
        );

    }



    cudaEventRecord(stop);



    cudaEventSynchronize(stop);



    float ms = 0.0F;



    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );



    cudaEventDestroy(start);

    cudaEventDestroy(stop);



    return ms / REPEAT;

}







//======================================================
// Check Result
//======================================================


bool check_result(
    const std::vector<float>& result
)
{

    for(float value : result)
    {

        if(
            std::fabs(
                value - K
            )
            >
            1e-3
        )
        {
            return false;
        }

    }


    return true;

}








//======================================================
// Main
//======================================================


int main()
{

    std::cout
        << "GEMM Benchmark\n\n";



    std::vector<float> h_A(
        M*K,
        1.0F
    );


    std::vector<float> h_B(
        K*N,
        1.0F
    );



    std::vector<float> h_C_naive(
        M*N
    );


    std::vector<float> h_C_tiled(
        M*N
    );


    std::vector<float> h_C_cublas(
        M*N
    );




    float* d_A=nullptr;

    float* d_B=nullptr;


    float* d_C_naive=nullptr;

    float* d_C_tiled=nullptr;

    float* d_C_cublas=nullptr;





    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            M*K*sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            K*N*sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_C_naive,
            M*N*sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_C_tiled,
            M*N*sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_C_cublas,
            M*N*sizeof(float)
        )
    );





    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            M*K*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );



    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            K*N*sizeof(float),
            cudaMemcpyHostToDevice
        )
    );





    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );


    dim3 grid(
        (N+TILE_SIZE-1)
        /
        TILE_SIZE,

        (M+TILE_SIZE-1)
        /
        TILE_SIZE
    );





    cublasHandle_t handle;


    CUBLAS_CHECK(
        cublasCreate(
            &handle
        )
    );




    float naive_ms =
        benchmark_naive(
            d_A,
            d_B,
            d_C_naive,
            grid,
            block
        );



    float tiled_ms =
        benchmark_tiled(
            d_A,
            d_B,
            d_C_tiled,
            grid,
            block
        );



    float cublas_ms =
        benchmark_cublas(
            handle,
            d_A,
            d_B,
            d_C_cublas
        );





    cudaMemcpy(
        h_C_naive.data(),
        d_C_naive,
        M*N*sizeof(float),
        cudaMemcpyDeviceToHost
    );


    cudaMemcpy(
        h_C_tiled.data(),
        d_C_tiled,
        M*N*sizeof(float),
        cudaMemcpyDeviceToHost
    );


    cudaMemcpy(
        h_C_cublas.data(),
        d_C_cublas,
        M*N*sizeof(float),
        cudaMemcpyDeviceToHost
    );





    std::cout
        << std::boolalpha;


    std::cout
        << "Naive correct: "
        << check_result(h_C_naive)
        << "\n";


    std::cout
        << "Tiled correct: "
        << check_result(h_C_tiled)
        << "\n";


    std::cout
        << "cuBLAS correct: "
        << check_result(h_C_cublas)
        << "\n\n";




    std::cout
        << std::fixed
        << std::setprecision(4);



    std::cout
        << "Naive GEMM : "
        << naive_ms
        << " ms   "
        << calculate_gflops(naive_ms)
        << " GFLOPS\n";


    std::cout
        << "Tiled GEMM : "
        << tiled_ms
        << " ms   "
        << calculate_gflops(tiled_ms)
        << " GFLOPS\n";


    std::cout
        << "cuBLAS GEMM: "
        << cublas_ms
        << " ms   "
        << calculate_gflops(cublas_ms)
        << " GFLOPS\n";





    cublasDestroy(handle);



    cudaFree(d_A);

    cudaFree(d_B);

    cudaFree(d_C_naive);

    cudaFree(d_C_tiled);

    cudaFree(d_C_cublas);



    return 0;

}
