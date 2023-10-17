#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

__device__ int* reduction_5(int *g_idata, int *g_odata);
__device__ void warpReduce(volatile int* sdata, int tid);
__device__ int* reduction_5(int *g_idata, int *g_odata)
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s = blockDim.x/2; s>32; s>>=1)
    {
        if(tid < s)
        {
          sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
    }
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