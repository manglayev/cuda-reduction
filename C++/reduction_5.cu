#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

__device__ void warpReduce(volatile int* sdata, int tid);
__device__ int* reduction_5(int *g_idata, int *g_odata)
{
    static __shared__ int sdata[THREADS];
    // each thread loads one element from global to shared mem
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[threadIdx.x] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s = blockDim.x/2; s>32; s>>=1)
    {
        if(threadIdx.x < s)
        {
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce(sdata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    //implement second reduction for the summed array
    __syncthreads();
    i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[threadIdx.x] = g_odata[i] + g_odata[i+blockDim.x];
    __syncthreads();
    for(unsigned int s = blockDim.x/2; s>32; s>>=1)
    {
        if(threadIdx.x < s)
        {
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce(sdata, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    return g_odata;
}

__device__ int* reduction_51(int *g_idata, int *g_odata)
{
    static __shared__ int sdata[THREADS];
    // each thread loads one element from global to shared mem
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[threadIdx.x] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s = blockDim.x/2; s>32; s>>=1)
    {
        if(threadIdx.x < s)
        {
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce(sdata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    return g_odata;
}

__device__ int* reduction_52(int *g_idata, int *g_odata)
{
    __shared__ int sdata[BLOCKS/4];
    // each thread loads one element from global to shared memory
    int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    sdata[threadIdx.x] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared memory
    for(unsigned int s = BLOCKS/8; s>32; s>>=1)
    {
        if (threadIdx.x < s)
        {
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce(sdata, threadIdx.x);
    // write result for this block to global memory
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    __syncthreads();
    return g_odata;
}

__device__ void warpReduce(volatile int* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
