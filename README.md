# CUDA High Performance Kernels


CUDA C++ implementation and optimization of GPU kernels for AI inference and high-performance computing.

This repository contains CUDA kernel implementations covering GPU programming fundamentals, memory optimization, parallel algorithms, matrix computation, deep learning operators and runtime optimization.


---

# Environment

- OS: Ubuntu 22.04 (WSL2)
- CUDA: 13.1
- Compiler: nvcc
- GPU: RTX 5070 Laptop GPU


---

# Project Structure


## 01. Basic CUDA Programming

Fundamental CUDA programming models and GPU execution basics.

Implemented:

- Hello CUDA
- Vector Add
- Matrix Add
- ReLU
- CUDA Event Timing


Key Concepts:

- Kernel launch
- Grid / Block / Thread hierarchy
- Thread indexing
- GPU memory allocation
- CPU-GPU synchronization


---

## 02. GPU Memory Optimization

Optimization techniques based on GPU memory hierarchy.

Implemented:

- CUDA Execution Model
- Thread Mapping
- Global Memory Access Pattern
- Memory Coalescing
- Naive Matrix Transpose
- Shared Memory Tiled Transpose
- Bank Conflict Analysis
- Padding Optimization


Optimization Techniques:

- Improve global memory access efficiency
- Reduce uncoalesced memory access
- Utilize shared memory data reuse
- Avoid shared memory bank conflicts


---

## 03. Parallel Algorithm Optimization

Implementation and optimization of parallel reduction algorithms.

Implemented:

- Naive Reduction
- Shared Memory Reduction
- Warp Shuffle Reduction
- Multi Block Reduction


Optimization Evolution:

