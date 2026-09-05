#include <cublas_v2.h>

cublasHandle_t handle;

cublasCreate(&handle);

const float alpha = 1.0F;

const float beta = 0.0F;

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
);

cublasDestroy(handle);
