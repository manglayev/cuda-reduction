import numpy as np
import numba
from numba import cuda

BLOCKS = 8
THREADS = 512
VARIANT = 1

def callReduction(a, b):
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)
    if VARIANT == 1:
        reduction_1[BLOCKS, THREADS](dev_a, dev_b)
    sum = dev_b.copy_to_host()
    return sum

@numba.cuda.jit
def reduction_1(g_idata, g_odata):
    sdata = numba.cuda.shared.array(shape=THREADS, dtype=numba.int32)
    i = numba.cuda.grid(1)
    sdata[numba.cuda.threadIdx.x] = g_idata[i]
    numba.cuda.syncthreads()
    #do reduction in shared memory
    s = 1
    while s < numba.cuda.blockDim.x:
        if numba.cuda.threadIdx.x % (2*s) == 0:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + s]
        numba.cuda.syncthreads()
        s*=2
    numba.cuda.syncthreads()
    # write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = sdata[0]
    #implement second reduction for the summed array
    numba.cuda.syncthreads()
    s = 1
    while s < numba.cuda.blockDim.x:
        if numba.cuda.threadIdx.x % (2*s) == 0:
            g_odata[cuda.threadIdx.x] += g_odata[numba.cuda.threadIdx.x + s]
        numba.cuda.syncthreads()
        s*=2
    numba.cuda.syncthreads()
    #write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = g_odata[0];
    numba.cuda.syncthreads()

if __name__ == "__main__":
    a = np.ones(BLOCKS*THREADS, dtype=np.int32)
    b = np.ones(1, dtype=np.int32)
    b = callReduction(a, b)
    print("GPU RESULTS: ", b[0])
    sum = np.sum(a)
    print("CPU RESULTS: ", sum)
