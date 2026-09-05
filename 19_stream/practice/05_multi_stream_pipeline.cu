for (int batch = 0; batch < NUM_BATCHES; ++batch) {

    const int stream_id = batch % NUM_BATCHES;

    const std::size_t offset =
        static_cast<std::size_t>(batch) * CHUNK_N;

    cudaMemcpyAsync(
        d_A[stream_id],
        h_A + offset,
        chunk_bytes,
        cudaMemcpyHostToDevice,
        streams[stream_id]
    );

    cudaMemcpyAsync(
        d_B[stream_id],
        h_B + offset,
        chunk_bytes,
        cudaMemcpyHostToDevice,
        streams[stream_id]
    );

    vetor_add_kernel<<<
        blocks,
        BLOCK_SIZE,
        0,
        streams[stream_id]
    >>>(
        d_A[stream_id],
        d_B[stream_id],
        d_C[stream_id],
        CHUNK_N
    );

    cudaMemcpyAsync(
        h_C + offset,
        d_C[stream_id],
        chunk_bytes,
        cudaMemcpyDeviceToHost,
        streams[stream_id]
    );
}

for (int s = 0; s < NUM_STREAMS; ++s) {

    cudaStreamSynchronize(stream[s]);
}
