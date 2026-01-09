#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#define SELECT_BLOCK_SIZE(n) ((n) > 50000000 ? 512 : 256)
//1.121888 ms

template <int blockSize>
__global__ void histgram(int *hist_data, int *bin_data, int N)
{
    __shared__ int cache[256];
    int gtid = blockIdx.x * blockSize + threadIdx.x; // 泛指当前线程在所有block范围内的全局id
    int tid = threadIdx.x; // 泛指当前线程在其block内的id
    cache[tid] = 0;
    __syncthreads();

    // for循环来自动确定每个线程处理的元素个数
    for (int i = gtid; i < N; i += gridDim.x * blockSize)
    {
        int val = hist_data[i];// 每个单线程计算全局内存中的若干个值
        atomicAdd(&cache[val], 1); // 原子加法，强行使得并行的CUDA线程串行执行加法，但是并不能保证顺序
    }
    __syncthreads();//此刻每个block的bin都已统计在cache这个smem中
    
    // 写回全局内存（blockSize=256，所以所有线程都需要写回）
    atomicAdd(&bin_data[tid], cache[tid]);
}

bool CheckResult(int *out, int* groudtruth, int N){
    for (int i = 0; i < N; i++){
        if (out[i] != groudtruth[i]) {
            printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groudtruth[i]);
            return false;
        }
    }
    return true;
}

int main(){
    float milliseconds = 0;
    const int N = 25600000;
    int *hist = (int *)malloc(N * sizeof(int));
    int *bin = (int *)malloc(256 * sizeof(int));
    int *bin_data;
    int *hist_data;
    cudaMalloc((void **)&bin_data, 256 * sizeof(int));
    cudaMalloc((void **)&hist_data, N * sizeof(int));

    for(int i = 0; i < N; i++){
        hist[i] = i % 256;
    }

    int *groudtruth = (int *)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 100000;
    }

    cudaMemcpy(hist_data, hist, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    //动态定义blockSize大小
    const int blockSize = SELECT_BLOCK_SIZE(N);
    //当有shared mem操作时， N / (GridSize × blockSize) < 2时，代表每个线程只处理一个元素，
    //此时block数量过多，gpu调度开销大，gpu调度为主要瓶颈，建议减少block数量
    //以1000万数据量为界，根据数据量动态调整每个线程要处理的元素个数（1~8）
    int elements_per_thread = min(8, max(1, (int)(N / 10000000.0)));
    int GridSize = std::min((N + blockSize * elements_per_thread - 1) / (blockSize * elements_per_thread), 
                            deviceProp.maxGridSize[0]);
    printf("elements_per_thread=%d, GridSize=%d, blockSize=%d\n", elements_per_thread, GridSize, blockSize);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // bug1: L68的N不能传错，之前传的256，导致L19的cache[1]打印出来为0
    histgram<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    // bug2: 同bug1，L67传进去的256表示两个buffer的数据量，这个必须得精确，之前传的N，尽管只打印第1个值，但依然导致L27打印出来的值为垃圾值
    bool is_right = CheckResult(bin, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%d ", bin[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);    

    cudaFree(bin_data);
    cudaFree(hist_data);
    free(bin);
    free(hist);
}
