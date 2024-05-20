#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

__device__ int* reduction_41(int *g_idata, int *g_odata)
{
    __shared__ int sdata[THREADS];
    // each thread loads one element from global to shared mem
    int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;
    sdata[threadIdx.x] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared memory
    for(unsigned int s = blockDim.x/2; s>0; s>>=1)
    {
        if (threadIdx.x < s)
        {
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    __syncthreads();
    return g_odata;
}