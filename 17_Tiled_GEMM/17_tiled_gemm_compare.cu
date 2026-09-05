#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess) {                           \
            std::cerr                                         \
                << "CUDA error: "                             \
                << cudaGetErrorString(error)                  \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << '\n';                                      \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


//============================================================
// 为了学习方便先使用2
//
// 后面真正做性能测试时可以改成：
// 16
//============================================================

constexpr int TILE_SIZE =
    2;


//============================================================
// Naive GEMM
//
// A: M x K
// B: K x N
// C: M x N
//
// One Thread computes one C[row][col]
//============================================================

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
        blockIdx.y
        *
        blockDim.y
        +
        threadIdx.y;


    const int col =
        blockIdx.x
        *
        blockDim.x
        +
        threadIdx.x;


    if (
        row >= M
        ||
        col >= N
    )
    {
        return;
    }


    float sum =
        0.0F;


    for (
        int k = 0;
        k < K;
        ++k
    )
    {
        sum +=
            A[
                row * K
                +
                k
            ]
            *
            B[
                k * N
                +
                col
            ];
    }


    C[
        row * N
        +
        col
    ]
        =
        sum;
}


//============================================================
// Tiled GEMM
//
// A / B Tile:
// Global Memory
//        ↓
// Shared Memory
//        ↓
// Block内重复使用
//
// One Thread still computes one C[row][col]
//============================================================

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
    //========================================================
    // 每个Block都有自己的一份Shared Memory
    //========================================================

    __shared__
    float shared_A
        [TILE_SIZE]
        [TILE_SIZE];


    __shared__
    float shared_B
        [TILE_SIZE]
        [TILE_SIZE];


    //========================================================
    // Thread在当前Block中的位置
    //========================================================

    const int tx =
        threadIdx.x;


    const int ty =
        threadIdx.y;


    //========================================================
    // 当前Thread最终负责的C[row][col]
    //========================================================

    const int row =
        blockIdx.y
        *
        TILE_SIZE
        +
        ty;


    const int col =
        blockIdx.x
        *
        TILE_SIZE
        +
        tx;


    //========================================================
    // 当前Thread自己的累加器
    //
    // 概念上通常放在Register
    //========================================================

    float sum =
        0.0F;


    //========================================================
    // 沿K维需要多少个Tile
    //
    // ceil(K / TILE_SIZE)
    //========================================================

    const int num_phases =
        (
            K
            +
            TILE_SIZE
            -
            1
        )
        /
        TILE_SIZE;


    //========================================================
    // 沿K方向一块一块推进
    //========================================================

    for (
        int phase = 0;
        phase < num_phases;
        ++phase
    )
    {
        //====================================================
        // 当前Phase中：
        //
        // A的K维是column
        //
        // B的K维是row
        //====================================================

        const int a_col =
            phase
            *
            TILE_SIZE
            +
            tx;


        const int b_row =
            phase
            *
            TILE_SIZE
            +
            ty;


        //====================================================
        // Load A Tile
        //
        // A:
        // M x K
        //
        // valid:
        // row < M
        // a_col < K
        //====================================================

        if (
            row < M
            &&
            a_col < K
        )
        {
            shared_A[ty][tx]
                =
                A[
                    row * K
                    +
                    a_col
                ];
        }
        else
        {
            // 越界位置补0
            shared_A[ty][tx]
                =
                0.0F;
        }


        //====================================================
        // Load B Tile
        //
        // B:
        // K x N
        //
        // valid:
        // b_row < K
        // col < N
        //====================================================

        if (
            b_row < K
            &&
            col < N
        )
        {
            shared_B[ty][tx]
                =
                B[
                    b_row * N
                    +
                    col
                ];
        }
        else
        {
            // 越界位置补0
            shared_B[ty][tx]
                =
                0.0F;
        }


        //====================================================
        // 所有Thread必须先完成Tile加载
        //====================================================

        __syncthreads();


        //====================================================
        // 当前Tile内部做点积
        //
        // A Tile的一行
        //
        // ×
        //
        // B Tile的一列
        //====================================================

        for (
            int k = 0;
            k < TILE_SIZE;
            ++k
        )
        {
            sum +=
                shared_A[ty][k]
                *
                shared_B[k][tx];
        }


        //====================================================
        // 保证所有Thread已经使用完当前Tile
        //
        // 下一Phase才能覆盖Shared Memory
        //====================================================

        __syncthreads();
    }


    //========================================================
    // 所有Phase都完成后
    //
    // 只有合法Thread写C
    //========================================================

    if (
        row < M
        &&
        col < N
    )
    {
        C[
            row * N
            +
            col
        ]
            =
            sum;
    }
}


//============================================================
// CPU Reference
//============================================================

void gemm_cpu(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M,
    int N,
    int K
)
{
    for (
        int row = 0;
        row < M;
        ++row
    )
    {
        for (
            int col = 0;
            col < N;
            ++col
        )
        {
            float sum =
                0.0F;


            for (
                int k = 0;
                k < K;
                ++k
            )
            {
                sum +=
                    A[
                        row * K
                        +
                        k
                    ]
                    *
                    B[
                        k * N
                        +
                        col
                    ];
            }


            C[
                row * N
                +
                col
            ]
                =
                sum;
        }
    }
}


//============================================================
// Check correctness
//============================================================

bool check_result(
    const std::vector<float>& reference,
    const std::vector<float>& result
)
{
    const int n =
        static_cast<int>(
            reference.size()
        );


    for (
        int idx = 0;
        idx < n;
        ++idx
    )
    {
        const float diff =
            std::fabs(
                reference[idx]
                -
                result[idx]
            );


        if (
            diff
            >
            1e-5F
        )
        {
            std::cerr
                << "Mismatch at idx = "
                << idx
                << '\n';


            std::cerr
                << "CPU = "
                << reference[idx]
                << '\n';


            std::cerr
                << "GPU = "
                << result[idx]
                << '\n';


            return false;
        }
    }


    return true;
}


//============================================================
// Print Matrix
//============================================================

void print_matrix(
    const std::vector<float>& matrix,
    int rows,
    int cols
)
{
    for (
        int row = 0;
        row < rows;
        ++row
    )
    {
        for (
            int col = 0;
            col < cols;
            ++col
        )
        {
            std::cout
                << matrix[
                    row * cols
                    +
                    col
                ]
                << '\t';
        }


        std::cout
            << '\n';
    }
}


//============================================================
// main
//============================================================

int main()
{
    //========================================================
    // 故意使用不能完全整除Tile的尺寸
    //
    // A: 5 x 7
    //
    // B: 7 x 6
    //
    // C: 5 x 6
    //========================================================

    const int M =
        5;


    const int K =
        7;


    const int N =
        6;


    //========================================================
    // Number of Elements
    //========================================================

    const int size_A =
        M
        *
        K;


    const int size_B =
        K
        *
        N;


    const int size_C =
        M
        *
        N;


    //========================================================
    // Bytes
    //========================================================

    const std::size_t bytes_A =
        static_cast<std::size_t>(
            size_A
        )
        *
        sizeof(float);


    const std::size_t bytes_B =
        static_cast<std::size_t>(
            size_B
        )
        *
        sizeof(float);


    const std::size_t bytes_C =
        static_cast<std::size_t>(
            size_C
        )
        *
        sizeof(float);


    //========================================================
    // Host Memory
    //========================================================

    std::vector<float>
        h_A(
            size_A
        );


    std::vector<float>
        h_B(
            size_B
        );


    std::vector<float>
        h_cpu_C(
            size_C
        );


    std::vector<float>
        h_naive_C(
            size_C
        );


    std::vector<float>
        h_tiled_C(
            size_C
        );


    //========================================================
    // Initialize A
    //
    // A = 1,2,3,...
    //========================================================

    for (
        int idx = 0;
        idx < size_A;
        ++idx
    )
    {
        h_A[idx] =
            static_cast<float>(
                idx + 1
            );
    }


    //========================================================
    // Initialize B
    //
    // 全部设成1
    //
    // 这样结果非常容易手算：
    //
    // C每一行
    // =
    // A对应行元素之和
    //========================================================

    for (
        int idx = 0;
        idx < size_B;
        ++idx
    )
    {
        h_B[idx] =
            1.0F;
    }


    //========================================================
    // CPU Reference
    //========================================================

    gemm_cpu(
        h_A,
        h_B,
        h_cpu_C,
        M,
        N,
        K
    );


    //========================================================
    // Device Pointers
    //========================================================

    float* d_A =
        nullptr;


    float* d_B =
        nullptr;


    float* d_naive_C =
        nullptr;


    float* d_tiled_C =
        nullptr;


    //========================================================
    // cudaMalloc
    //========================================================

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            bytes_A
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            bytes_B
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_naive_C,
            bytes_C
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_tiled_C,
            bytes_C
        )
    );


    //========================================================
    // Host -> Device
    //========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            bytes_A,
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            bytes_B,
            cudaMemcpyHostToDevice
        )
    );


    //========================================================
    // Block / Grid
    //
    // TILE_SIZE=2:
    //
    // block:
    // 2 x 2
    //
    // grid.x:
    // ceil(N / 2)
    //
    // grid.y:
    // ceil(M / 2)
    //========================================================

    dim3 block(
        TILE_SIZE,
        TILE_SIZE
    );


    dim3 grid(
        (
            N
            +
            TILE_SIZE
            -
            1
        )
        /
        TILE_SIZE,

        (
            M
            +
            TILE_SIZE
            -
            1
        )
        /
        TILE_SIZE
    );


    //========================================================
    // Naive GEMM
    //========================================================

    naive_gemm_kernel<<<
        grid,
        block
    >>>(
        d_A,
        d_B,
        d_naive_C,
        M,
        N,
        K
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //========================================================
    // Tiled GEMM
    //========================================================

    tiled_gemm_kernel<<<
        grid,
        block
    >>>(
        d_A,
        d_B,
        d_tiled_C,
        M,
        N,
        K
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    //========================================================
    // 等所有Kernel结束
    //========================================================

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    //========================================================
    // Device -> Host
    //========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_naive_C.data(),
            d_naive_C,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_tiled_C.data(),
            d_tiled_C,
            bytes_C,
            cudaMemcpyDeviceToHost
        )
    );


    //========================================================
    // Correctness
    //========================================================

    const bool naive_correct =
        check_result(
            h_cpu_C,
            h_naive_C
        );


    const bool tiled_correct =
        check_result(
            h_cpu_C,
            h_tiled_C
        );


    //========================================================
    // Print
    //========================================================

    std::cout
        << std::boolalpha;


    std::cout
        << "M = "
        << M
        << '\n';


    std::cout
        << "N = "
        << N
        << '\n';


    std::cout
        << "K = "
        << K
        << '\n';


    std::cout
        << "TILE_SIZE = "
        << TILE_SIZE
        << '\n';


    std::cout
        << "grid = ("
        << grid.x
        << ", "
        << grid.y
        << ")\n";


    std::cout
        << "block = ("
        << block.x
        << ", "
        << block.y
        << ")\n";


    std::cout
        << '\n';


    std::cout
        << "Naive GEMM correct = "
        << naive_correct
        << '\n';


    std::cout
        << "Tiled GEMM correct = "
        << tiled_correct
        << '\n';


    std::cout
        << "\nCPU Reference C:\n";


    print_matrix(
        h_cpu_C,
        M,
        N
    );


    std::cout
        << "\nTiled GPU C:\n";


    print_matrix(
        h_tiled_C,
        M,
        N
    );


    //========================================================
    // Free Device Memory
    //========================================================

    CUDA_CHECK(
        cudaFree(
            d_A
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_B
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_naive_C
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_tiled_C
        )
    );


    return 0;
}
