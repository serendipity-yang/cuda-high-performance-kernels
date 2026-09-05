cudaStream_t stream0;
cudaStream_t stream1;

cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

cudaEvent_t event;

cudaEventCreateWithFlags(
    &event,
    cudaEventDisableTiming
);

kernelA<<<
    blocks,
    threads,
    0,
    stream0
>>>();

cudaEventRecord(
    event,
    stream0
);

cudaStreamWaitEvent(
    stream1,
    event,
    0
);

kernelB<<<
    blocks,
    threads,
    0,
    stream1
>>>();
