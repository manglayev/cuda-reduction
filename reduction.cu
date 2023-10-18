#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "reduction_1.cu"
#include "reduction_2.cu"
#include "reduction_3.cu"
#include "reduction_4.cu"
#include "reduction_5.cu"
#include "reduction_6.cu"
#include "reduction_7.cu"

__global__ void cuda_global(int *dev_a, int *dev_b)
{
  switch (VARIANT)
  {
    case 1:
      dev_b = reduction_1(dev_a, dev_b);
      break;
    case 2:
      dev_b = reduction_2(dev_a, dev_b);
      break;
    case 3:
      dev_b = reduction_3(dev_a, dev_b);
      break;
    case 4:
      dev_b = reduction_4(dev_a, dev_b);
      break;
    case 5:
      dev_b = reduction_5(dev_a, dev_b);
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

  int a[CUDASIZE] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int b[1];

  int *dev_a;
  int *dev_b;

  cudaMalloc((void**)&dev_a, CUDASIZE*sizeof(int));
  cudaMalloc((void**)&dev_b, sizeof(int));

  cudaMemcpy(dev_a, a, CUDASIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, sizeof(int), cudaMemcpyHostToDevice);

  cuda_global<<<BLOCKS, THREADS>>>(dev_a, dev_b);

  cudaMemcpy(b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);
  printf("GPU RESULTS: b = %d\n", b[0]);
  int sum = checkResults(a);
  printf("CPU RESULTS: sum = %d\n", sum);

  cudaFree(dev_a);
  cudaFree(dev_b);
  printf("STAGE 3 WRAPPER END\n");
}
