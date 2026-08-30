# Warp Shuffle Reduction Benchmark


## Environment


GPU:

RTX 5070 Laptop


CUDA:

13.1



## Test Configuration


Input:

N =


Block size:

256



## Results


| Kernel | Time(ms) |
|---|---|
| Naive Reduction | |
| Shared Memory Reduction | |
| Warp Shuffle Reduction | |



## Observation


Warp shuffle reduces the overhead of shared memory communication inside a warp.

