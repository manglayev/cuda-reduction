#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"


template<unsigned int blockSize>
__device__ int* reduction_6(int *g_idata, int *g_odata)
{
    static __shared__ int sdata[THREADS];
    // each thread loads one element from global to shared mem
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[threadIdx.x] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (threadIdx.x < 256)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (threadIdx.x < 64)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
        warpReduce<blockSize>(sdata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    //implement second reduction for the summed array
    if (threadIdx.x < 32)
        warpReduce<blockSize>(g_odata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0) 
    {
        g_odata[blockIdx.x] = g_odata[0];
    }    
    return g_odata;
}