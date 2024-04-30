import numpy as np
from numba import cuda, int32

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

@cuda.jit
def reduction_1(g_idata, g_odata):
    sdata = cuda.shared.array(shape=THREADS, dtype=int32)
    i = cuda.grid(1)
    sdata[cuda.threadIdx.x] = g_idata[i]
    cuda.syncthreads()
    #do reduction in shared mem
    s = 1
    while s < cuda.blockDim.x:
        if cuda.threadIdx.x % (2*s) == 0:
            sdata[cuda.threadIdx.x] += sdata[cuda.threadIdx.x + s]
        cuda.syncthreads()
        s*=2
    # write result for this block to global mem
    if cuda.threadIdx.x == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]
    #implement second reduction for the summed array
    s = 1
    while s < cuda.blockDim.x:
        if cuda.threadIdx.x % (2*s) == 0:
            sdata[cuda.threadIdx.x] += g_odata[cuda.threadIdx.x + s]
        cuda.syncthreads()
        s*=2
    #write result for this block to global mem
    if cuda.threadIdx.x == 0:
        g_odata[cuda.blockIdx.x] = g_odata[0];

if __name__ == "__main__":
    a = np.ones(BLOCKS*THREADS, dtype=np.int32)
    b = np.ones(1, dtype=np.int32)
    b = callReduction(a, b)
    print("GPU RESULTS: ", b[0])
    sum = np.sum(a)
    print("CPU RESULTS: ", sum)
