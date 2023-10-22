#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

__device__ int* reduction_2(int *g_idata, int *g_odata)
{
  __shared__ int sdata[THREADS];
  // each thread loads one element from global to shared mem
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = g_idata[i];
  __syncthreads();
  // do reduction in shared mem
  for(unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    int index = 2 * s * threadIdx.x;
    if (index < blockDim.x)
    {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (threadIdx.x == 0)
  {
    g_odata[blockIdx.x] = sdata[0];
  }
  //implement second reduction for the summed array
  for(unsigned int s = 1; s < blockDim.x; s *= 2)
  {
    int index = 2 * s * threadIdx.x;
    if (index < blockDim.x)
    {
      g_odata[index] += g_odata[index + s];
    }
    __syncthreads();
  }
 // write result for this block to global mem
  if (threadIdx.x == 0) 
  {
    g_odata[blockIdx.x] = g_odata[0];
  }
  return g_odata;
}