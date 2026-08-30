# Shared Memory Reduction 学习笔记


## 1. 实验目的


本实验学习如何利用 CUDA Shared Memory 优化 Reduction。


Reduction（归约）：

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


Reduction是深度学习中非常常见的操作：

- Softmax
- LayerNorm
- Attention
- BatchNorm


---

# 2. Reduction为什么需要优化


CPU中：

```cpp
float sum = 0;

for(int i=0;i<n;i++)
{
    sum += input[i];
}
```


只有一个线程执行。


GPU希望：

多个线程并行处理。


例如：

```
input:

1 2 3 4 5 6 7 8


thread0

thread1

thread2

thread3

...

```


但是：

多个线程计算完成以后，需要把结果合并。


这个过程就是Reduction。


---

# 3. Naive Reduction的问题


最简单的方法：

线程直接访问Global Memory。


流程：

```
Global Memory

      |

      v

Thread

      |

      v

Partial Sum

```


问题：

## 3.1 Global Memory访问慢


Global Memory位于GPU显存。


访问延迟较高。


---

## 3.2 多线程通信效率低


线程之间需要共享计算结果。


如果频繁访问Global Memory：

性能下降。


---

# 4. 为什么引入Shared Memory


Shared Memory：

是GPU SM内部高速缓存。


位置：

```
GPU SM

 |
 |
Shared Memory

 |
 |
CUDA Threads

```


特点：

- 延迟低
- block内部线程共享
- 适合线程通信


---

# 5. Shared Memory Reduction基本思想


第一步：

每个线程从Global Memory读取数据。


```
Global Memory

thread0

thread1

thread2

...

```


第二步：

存入Shared Memory。


```cpp
__shared__ float sdata[];
```


结构：

```
Shared Memory


sdata[0]

sdata[1]

sdata[2]

...


```


第三步：

多个线程在Shared Memory中进行归约。


---

# 6. Block内部Reduction过程


假设：

block有8个线程。


初始：


```
thread0  1
thread1  2
thread2  3
thread3  4
thread4  5
thread5  6
thread6  7
thread7  8

```


第一次：

stride=4


```
thread0 += thread4

thread1 += thread5

thread2 += thread6

thread3 += thread7

```


结果：


```
thread0 6
thread1 8
thread2 10
thread3 12

```


继续：


stride=2


```
thread0 += thread2

thread1 += thread3

```


最后：


```
thread0 = final sum

```


---

# 7. 为什么需要__syncthreads()


代码中经常看到：

```cpp
__syncthreads();
```


作用：

让block中的所有线程同步。


例如：

线程0正在读取：

```
sdata[4]
```


但是线程4还没有写入。


如果不同步：

读取错误数据。


所以：

每次Shared Memory更新以后：

需要：

```
写入

↓

同步

↓

继续计算

```


---

# 8. Block和Thread映射


本实验：

每个block负责一部分输入。


例如：

输入：

```
N=1024

```


Block:

```
256 threads

```


那么：

一个block处理：

```
256个元素

```


计算：

```cpp
idx =
blockIdx.x * blockDim.x
+
threadIdx.x;
```


得到：

全局线程编号。


---

# 9. Shared Memory布局


一个block：

```
Block


thread0

thread1

thread2

...

thread255

```


Shared Memory：

```
sdata[0]

sdata[1]

...

sdata[255]

```


每个线程对应一个位置：

```cpp
sdata[threadIdx.x]
```


---

# 10. 多Block Reduction


GPU中：

不同block之间不能直接同步。


因此：

每个block先计算自己的partial sum。


例如：


```
Block0

sum0


Block1

sum1


Block2

sum2

```


最后：

```
sum0 + sum1 + sum2

=

final result

```


---

# 11. 实验结果


运行验证：


```bash
nvcc shared_memory_reduction.cu -o shared

./shared
```


结果：

```
CPU sum == GPU sum

correct = true
```


说明：

Shared Memory Reduction结果正确。


---

# 12. Shared Memory Reduction总结


通过本实验理解：


1. Global Memory访问速度较慢。


2. Shared Memory可以作为block内部高速通信区域。


3. Reduction可以利用树形结构减少计算次数。


4. Shared Memory需要使用__syncthreads保证线程同步。


5. Block之间不能直接通信，需要Multi-Block Reduction。


6. Warp Shuffle Reduction是在Shared Memory Reduction基础上的进一步优化。


