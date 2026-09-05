cudaMemcpy(
    d_A,
    h_A.data(),
    bytes,
    cudaMemcpyHostToDevice
);

cudaMemcpy(
    d_B,
    h_B.data(),
    bytes,
    cudaMemcpyHostToDevice
);

vector_add_kernel<<<
    blocks,
    BLOCK_SIZE
>>>(
    d_A,
    d_B,
    d_C,
    N
);

cudaDeviceSynchronize();

cudaMemcpy(
    h_C.data(),
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
);
