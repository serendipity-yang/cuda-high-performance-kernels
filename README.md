# CUDA High Performance Kernels

CUDA C++ implementation and optimization practice for GPU computing.

This repository contains CUDA kernel implementations and performance optimization experiments, including memory optimization, parallel algorithms, matrix computation and asynchronous execution.

---

## Environment

- OS: Ubuntu 22.04 (WSL2)
- CUDA: 13.1
- Compiler: nvcc
- GPU: RTX 5070 Laptop GPU


---

## Contents


## 01. Basic CUDA Programming

- Hello CUDA
- Vector Add
- Matrix Add
- ReLU


## 02. GPU Memory Optimization

- CUDA Execution Model
- Global Memory Access Pattern
- Memory Coalescing
- Shared Memory
- Bank Conflict
- Padding Optimization


## 03. Parallel Algorithm Optimization

- Naive Reduction
- Shared Memory Reduction
- Warp Shuffle Reduction


## 04. Matrix Computation

- Naive GEMM
- Tiled GEMM
- cuBLAS Comparison


## 05. Deep Learning Operators

- Softmax
- LayerNorm
- Activation Function
- Operator Fusion


## 06. CUDA Runtime Optimization

- CUDA Stream
- Async Memory Copy
- Pinned Memory
- Pipeline Execution


---

## Optimization Topics

- Memory Coalescing
- Shared Memory Optimization
- Warp-Level Programming
- Reduction Optimization
- Tile-Based Computation
- Kernel Fusion


---

## Future Work

- Nsight Systems Analysis
- Nsight Compute Profiling
- Tensor Core Optimization
- FP16/INT8 Kernel Optimization
