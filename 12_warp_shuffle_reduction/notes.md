# Warp Shuffle Reduction 学习笔记


## 1. 实验目的


本实验学习如何使用 CUDA Warp Shuffle 优化 Reduction。


Reduction（归约）的目标：

将多个输入元素合并成一个结果。


例如：

输入：

```
[1,2,3,4]
```


输出：

```
10
```


在深度学习中，Reduction经常出现：

- Softmax
- LayerNorm
- Attention
- BatchNorm


---

# 2. 为什么GPU需要Reduction


CPU执行：

```cpp
float sum = 0;

for(int i=0;i<n;i++)
{
    sum += input[i];
}
```


特点：

一个线程顺序执行。


GPU希望：

多个线程同时计算。


例如：

输入：

```
[1 2 3 4 5 6 7 8]
```


多个线程：

```
thread0 -> 1
thread1 -> 2
thread2 -> 3
thread3 -> 4
...
```


但是：

最后需要把多个线程结果合并。


这个过程就是：

Reduction。


---

# 3. Reduction的核心问题


GPU线程之间如何通信？


例如：

线程计算：

```
thread0 = 1

thread1 = 2

```


需要：

```
thread0 = thread0 + thread1

```


线程之间必须交换数据。


CUDA提供：

- Global Memory
- Shared Memory
- Warp Shuffle


三种方式。


---

# 4. Naive Reduction


最简单方法：

线程直接从global memory读取数据。


流程：

```
Global Memory

      |

      v

Thread计算

      |

      v

Reduction

```


问题：

## 4.1 Global Memory访问慢


Global Memory位于GPU显存。


访问延迟较高。


---

## 4.2 同步开销


多个线程需要等待。


例如：

```
thread0
thread1

计算完成

等待

继续合并

```


效率较低。


---

# 5. Shared Memory Reduction


优化方法：

把数据先加载到Shared Memory。


流程：

```
Global Memory

      |

      v

Shared Memory

      |

      v

Reduction

```


原因：

Shared Memory位于SM内部。


速度：

```
Shared Memory

>

Global Memory
```


---

## Shared Memory布局


一个block：

```
Block


thread0

thread1

thread2

...

thread255

```


申请：

```cpp
__shared__ float sdata[];
```


每个线程把数据放入：

```
sdata[threadIdx.x]
```


然后block内部进行归约。


---

# 6. Warp Shuffle Reduction


进一步优化：


CUDA提供：

```cpp
__shfl_down_sync()
```


作用：

让同一个warp中的线程直接交换register数据。


不需要：

- Shared Memory
- __syncthreads()


---

# 7. 什么是Warp？


CUDA执行基本单位：

Warp。


一个warp固定：

```
32 threads
```


例如：

一个block：

```
256 threads
```


那么：

```
256 / 32

= 8 warps
```


所以：

一个block包含：

```
8个warp
```


---

# 8. __shfl_down_sync原理


假设warp：

```
thread0 value=1
thread1 value=2
thread2 value=3
thread3 value=4

```


执行：

```cpp
__shfl_down_sync(
    mask,
    value,
    offset
)
```


offset=1:


数据移动：

```
thread0 <- thread1

thread1 <- thread2

thread2 <- thread3

```


然后：

```cpp
value += other;
```


完成：

```
thread0 = 1+2

thread1 = 2+3

thread2 = 3+4

```


---

# 9. 本实验Kernel配置


输入：

```
N = 4194304
```


Block大小：

```
256 threads
```


因为：

```
warpSize = 32
```


所以：

```
256 / 32 = 8 warps
```


一个block：

```
8个warp
```


---

# 10. 多Block Reduction


一个GPU无法让所有线程直接通信。


因此：

多个block分别计算部分结果。


例如：


```
Block0

partial_sum0


Block1

partial_sum1


Block2

partial_sum2

```


最后：

```
partial_sum0

+

partial_sum1

+

partial_sum2


=

final result

```


---

# 11. 实验结果


运行：


```bash
nvcc warp_shuffle_reduction.cu -o warp

./warp
```


输出：


```
n = 4194304

threads per block = 256

blocks = 16384

CPU sum = 4.1943e+06

GPU sum = 4.1943e+06

correct = true

```


说明：

GPU Reduction结果正确。


---

# 12. 本次实验总结


通过本实验理解：


1. Reduction本质是多个数据合并。


2. GPU线程之间需要通信。


3. Shared Memory可以提高block内部通信效率。


4. Warp Shuffle可以进一步优化warp内部通信。


5. Warp是CUDA执行的重要单位，一个warp包含32个线程。


6. 多Block Reduction需要先计算partial sum，再合并最终结果。


