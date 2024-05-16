#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

__device__ int* reduction_30(int *g_idata, int *g_odata)
{
  __shared__ int sdata[THREADS];
  // each thread loads one element from global to shared mem
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s = blockDim.x/2; s > 0; s>>=1)
  {
    if (threadIdx.x < s)
    {
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (threadIdx.x == 0)
  {
    g_odata[blockIdx.x] = sdata[0];
  }
  __syncthreads();
  return g_odata;
}
