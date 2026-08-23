//1. 引入头文件
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

//2. 定义CUDA_CHECK宏
#define CUDA_CHECK(call)                            \
    do {                                            \
        cudaError_t error = (call);                 \
        if (error != cudaSuccess) {                 \
            std::cerr << "CUDA error:"              \
                      << cudaGetErrorString(error)  \
                      << " at " << __FILE__         \
                      << ':' << __LINE__            \
                      << '\n';                      \
            std::exit(EXIT_FAILURE);                \
        }                                           \
    } while (0)

//3. 定义ReLU Kernel
__global__ void relu(
    const float* input,
    float* output,
    int n) {

    int idx = 
        blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < n) {
        output[idx] = 
            input[idx] > 0.0F
            ? input[idx]
            : 0.0F;
    }
}

//4. 设置输入规模n和bytes
int main() {
    const int n = 10'000'000;

    const std::size_t bytes =
        static_cast<std::size_t>(n) *
        sizeof(float);

//5. 创建CPU输入与两份输出
    std::vector<float> h_input(n);
    std::vector<float> h_cpu_output(n);
    std::vector<float> h_gpu_output(n);

//6. 初始化输入数据
    for (int idx = 0; idx < n; ++idx) {
        const float value = 
            static_cast<float>(idx % 1000) /
            10.0F;

        h_input[idx] = 
            idx % 2 == 0
                ? value
                : -value;
    }

//7. steady_clock记录CPU开始时间
    const auto cpu_start = 
        std::chrono::steady_clock::now();

//8. CPU执行ReLU
    for (int idx = 0; idx < n; ++idx) {
        h_cpu_output[idx] = 
            h_input[idx] > 0.0F
                ? h_input[idx]
                : 0.0F;
    }

//9. steady_clock记录CPU结束时间
    const auto cpu_stop = 
        std::chrono::steady_clock::now();

//10. duration转换成毫秒并调用count
    const double cpu_ms = 
        std::chrono::duration<
        double,
        std::milli
        >(
            cpu_stop - cpu_start
        ).count();
    
    std::cout 
        << "element count = "
        << n
        << '\n'
        << "data size = "
        << static_cast<double>(bytes) /
                (1024.0 * 1024.0)
        << " MiB\n"
        << "CPU ReLU time = "
        << cpu_ms
        << " ms\n";
    
//11. 创建GPU指针
    float* d_input = nullptr;
    float* d_output = nullptr;

//12. 申请GPU显存
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_input),
        bytes
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_output),
        bytes
    ));

//13. HostToDevice复制输入
    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

//14. 配置blocks和threads
    const int threads_per_block = 256;

    const int blocks = 
        (n + threads_per_block - 1) /
        threads_per_block;

//15. 预热Kernel
    const int warmup_iterations = 10;

    for (int iteration = 0; 
        iteration < warmup_iterations;
        ++iteration) {

        relu<<<blocks,threads_per_block>>>(
            d_input,
            d_output,
            n
            );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());


//16. 创建gpu_start和gpu_stop事件
    cudaEvent_t gpu_start;
    cudaEvent_t gpu_stop;

    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_stop));


//17. 连续运行100次Kernel
    const int benchmark_iterations = 100;

//18. Event计算总时间
    CUDA_CHECK(cudaEventRecord(gpu_start));
    
    for (int iteration = 0;
        iteration < benchmark_iterations;
        ++iteration) {
        
        relu<<<blocks, threads_per_block>>>(
            d_input,
            d_output,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(gpu_stop));

    CUDA_CHECK(cudaEventSynchronize(gpu_stop));

//19. 计算平均Kernel时间
    float total_gpu_ms = 0.0F;
    
    CUDA_CHECK(cudaEventElapsedTime(
        &total_gpu_ms,
        gpu_start,
        gpu_stop
    ));

    const float average_gpu_ms = 
        total_gpu_ms /
        static_cast<float>(benchmark_iterations);

//20. DeviceToHost复制结果
    CUDA_CHECK(cudaMemcpy(
        h_gpu_output.data(),
        d_output,
        bytes,
        cudaMemcpyDeviceToHost
    ));

//21. 验证CPU与GPU结果
    const float tolerance = 1.0e-6F;

    bool all_correct = true;

    for (int idx = 0; idx < n; ++idx) {
        const float difference = 
            std::fabs(
                h_cpu_output[idx] -
                h_gpu_output[idx]
            );

        if (difference > tolerance) {
            all_correct = false;

            std::cerr
                << "Mismatch at idx = "
                << idx
                << ", CPU = "
                << h_cpu_output[idx]
                << ", GPU = "
                << h_gpu_output[idx]
                << '\n';

            break;

        }
    }

//22. 打印时间和加速比
    std::cout 
        << "GPU total time for "
        << benchmark_iterations
        << " iterations = "
        << total_gpu_ms
        << " ms\n"
        << "GPU average kernel time = "
        << average_gpu_ms
        << " ms\n"
        << "Verification = "
        << (all_correct ? "PASS" : "FAIL")
        << '\n';

    if (average_gpu_ms > 0.0F) {
        const double speed_ratio = 
            cpu_ms /
            static_cast<double>(average_gpu_ms);

        std::cout 
            << "CPU time / GPU average kernel time = "
            << speed_ratio
            << '\n';
    }


//23. 销毁Event
    CUDA_CHECK(cudaEventDestroy(gpu_start));
    CUDA_CHECK(cudaEventDestroy(gpu_stop));

//24. 释放显存
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

//25. 返回验证结果
    return all_correct
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}

