#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "reduction_warp.cu"
#include "reduction_1.cu"
#include "reduction_2.cu"
#include "reduction_3.cu"
#include "reduction_4.cu"
//#include "reduction_5.cu"
#include "reduction_6.cu"
#include "reduction_7.cu"

#include "reduction_10.cu"
#include "reduction_20.cu"
#include "reduction_30.cu"
#include "reduction_41.cu"
#include "reduction_42.cu"
#include "reduction_50.cu"

__global__ void cuda_global(int *dev_a, int *dev_b)
{
  switch (VARIANT)
  {
    case 1:
      //dev_b = reduction_1(dev_a, dev_b);
      dev_b = reduction_10(dev_a, dev_b);
      break;
    case 2:
      //dev_b = reduction_2(dev_a, dev_b);
      dev_b = reduction_20(dev_a, dev_b);
      break;
    case 3:
      //dev_b = reduction_3(dev_a, dev_b);
      dev_b = reduction_30(dev_a, dev_b);
      break;
    case 4:
      //dev_b = reduction_4(dev_a, dev_b);
      if(blockDim.x == THREADS)
        dev_b = reduction_41(dev_a, dev_b);
      if(blockDim.x == BLOCKS/4)
        dev_b = reduction_42(dev_a, dev_b);
      break;
    case 5:
      //dev_b = reduction_5(dev_a, dev_b);
      if(blockDim.x == THREADS)
        dev_b = reduction_51(dev_a, dev_b);
      if(blockDim.x == BLOCKS/4)
        dev_b = reduction_52(dev_a, dev_b);
      break;
    case 6:
      dev_b = reduction_6<THREADS>(dev_a, dev_b);
      break;
    case 7:
      dev_b = reduction_7<THREADS>(dev_a, dev_b);
      break;
    default:
      dev_b = reduction_1(dev_a, dev_b);
      break;
  }
}

int* initArray()
{
  static int array[CUDASIZE];
  for(int i = 0; i < CUDASIZE; i++)
  {
    array[i] = 1;
  }
  return array;
}

int checkResults(int *a)
{
  int sum = 0;
  for(int i = 0; i < CUDASIZE; i++)
  {
    sum = sum + a[i];
  }
  return sum;
}

void wrapper()
{
  printf("STAGE 3 WRAPPER START\n");
  /*
  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, 0);
  printf("  --- General information for device START ---\n");
  printf("Name: %s;\n", device.name);
  printf("Compute capability: %d.%d\n", device.major, device.minor);
  printf("Clock rate: %d\n", device.clockRate);
  printf("Total global memory: %ld\n", device.totalGlobalMem);
  printf("Total constant memory: %ld\n", device.totalConstMem);
  printf("Multiprocessor count: %d\n", device.multiProcessorCount);
  printf("Shared memory per block: %ld\n", device.sharedMemPerBlock);
  printf("Registers per block: %d\n", device.regsPerBlock);
  printf("Threads in warp: %d\n", device.warpSize);
  printf("Max threads Per Block: %d\n", device.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", device.maxThreadsDim[0], device.maxThreadsDim[1], device.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", device.maxGridSize[0], device.maxGridSize[1], device.maxGridSize[2]);
  printf("  --- General information for device END ---\n");
  */
  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  int *a = initArray();
  int b[1];
  int *dev_a;
  int *dev_b;

  cudaMalloc((void**)&dev_a, CUDASIZE*sizeof(int));
  cudaMemcpy(dev_a, a, CUDASIZE*sizeof(int), cudaMemcpyHostToDevice);

  if(VARIANT < 4)
  {
    //cudaMalloc((void**)&dev_b, sizeof(int));
    cudaMalloc((void**)&dev_b, BLOCKS*sizeof(int));
    //cudaMemcpy(dev_b, b, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_b, b, BLOCKS*sizeof(int), cudaMemcpyHostToDevice);
  }
  else
  {
    //cudaMalloc((void**)&dev_b, sizeof(int));
    cudaMalloc((void**)&dev_b, BLOCKS/2*sizeof(int));
    //cudaMemcpy(dev_b, b, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_b, b, BLOCKS/2*sizeof(int), cudaMemcpyHostToDevice);
  }

  switch(VARIANT)
  {
    case 1:
    {
      cuda_global<<<BLOCKS, THREADS>>>(dev_a, dev_b);
      cuda_global<<<1, BLOCKS>>>(dev_b, dev_b);
      break;
    }
    case 2:
    {
      cuda_global<<<BLOCKS, THREADS>>>(dev_a, dev_b);
      cuda_global<<<1, BLOCKS>>>(dev_b, dev_b);
      break;
    }
    case 3:
    {
      cuda_global<<<BLOCKS, THREADS>>>(dev_a, dev_b);
      cuda_global<<<1, BLOCKS>>>(dev_b, dev_b);
      break;
    }
    case 4:
    {
      cuda_global<<<BLOCKS/2, THREADS>>>(dev_a, dev_b);
      cuda_global<<<1, BLOCKS/4>>>(dev_b, dev_b);
      break;
    }
    case 5:
    {
      cuda_global<<<BLOCKS/2, THREADS>>>(dev_a, dev_b);
      cuda_global<<<1, BLOCKS/4>>>(dev_b, dev_b);
      break;
    }
    default:
    {
      cuda_global<<<BLOCKS, THREADS>>>(dev_a, dev_b);
      cuda_global<<<1, BLOCKS>>>(dev_b, dev_b);
      break;
    }
  }

  //cudaMemcpy(b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU RESULTS: VARIANT = %d; elapsed time: %.5f ms; \n", VARIANT, elapsedTime);
  printf("GPU RESULTS: sum = %d \n", b[0]);
  int sum = checkResults(a);
  printf("CPU RESULTS: sum = %d\n", sum);
  cudaFree(dev_a);
  cudaFree(dev_b);
  printf("STAGE 3 WRAPPER END\n");
}
