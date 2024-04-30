import numpy as np
from numba import cuda, int32

BLOCKS = 8
THREADS = 512
VARIANT = 1

def initializeArray(blocks_times_threads):
    return [1] * blocks_times_threads

def checkResults(array):
    sum = 0
    for element in array:
        sum += element
    return sum

def callReduction(dev_a, dev_b):
    if VARIANT == 1:
        reduction_1[BLOCKS, THREADS](dev_a, dev_b)

@cuda.jit
def reduction_1(g_idata, g_odata):
    sdata = cuda.shared.array(shape=THREADS, dtype=int32)
    i = cuda.grid(1)
    sdata[cuda.threadIdx.x] = g_idata[i]
    cuda.syncthreads()
    s = 1
    
    while s < cuda.blockDim.x:
        if cuda.threadIdx.x % (2*s) == 0:
            sdata[cuda.threadIdx.x] += sdata[cuda.threadIdx.x + s]
        s*=2
        cuda.syncthreads()
    #do reduction in shared mem
    '''
    for s in range(1, cuda.blockDim.x, s*2):
        if cuda.threadIdx.x % (2*s) == 0:
            sdata[cuda.threadIdx.x] += sdata[cuda.threadIdx.x + s]
        cuda.syncthreads()
    '''
    # write result for this block to global mem
    if cuda.threadIdx.x == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]
    #implement second reduction for the summed array
    s = 1
    while s < cuda.blockDim.x:
        if cuda.threadIdx.x % (2*s) == 0:
            sdata[cuda.threadIdx.x] += g_odata[cuda.threadIdx.x + s]
        s*=2
        cuda.syncthreads()
    '''
    for s in range(1, cuda.blockDim.x, s*2):
        if cuda.threadIdx.x % (2*s) == 0:
            g_odata[cuda.threadIdx.x] += g_odata[cuda.threadIdx.x + s];
        cuda.syncthreads();
    '''
    #write result for this block to global mem
    if cuda.threadIdx.x == 0:
        g_odata[cuda.blockIdx.x] = g_odata[0];

if __name__ == "__main__":
    a = initializeArray(BLOCKS*THREADS)
    b = initializeArray(BLOCKS*THREADS)

    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)

    callReduction(dev_a, dev_b)

    b = dev_b.copy_to_host()
    print("GPU RESULTS: b =", b[0])

    sum = checkResults(a)
    print("CPU RESULTS: sum =", sum)
