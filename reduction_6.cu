#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "reduction_warp.cu"

template<unsigned int blockSize>
__device__ int* reduction_6(int *g_idata, int *g_odata)
{
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    return g_odata;
}