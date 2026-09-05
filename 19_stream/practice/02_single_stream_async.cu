float* h_A = nullptr;

cudaMallocHost(&h_A, bytes);

cudaStream_t stream;

cudaStreamCreate(&stream);

cudaMemcpyAsync(
    d_A,
    h_A,
    bytes,
    cudaMemcpyHostToDevice,
    stream
);

cudaMemcpyAsync(
    d_B,
    h_B,
    bytes,
    cudaMemcpyHostToDevice,
    stream
);

kernel<<<
    blocks,
    BLOCK_SIZE,
    0,
    stream
>>>();

cudaMemcpyAsync(
    h_A,
    d_A,
    bytes,
    cudaMemcpyDeviceToHost,
    stream
);

cudaStreamSynchronize(stream);

cudaStreamDestroy(stream);

cudaFreeHost(h_A);
