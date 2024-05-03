from helpers import *

@numba.cuda.jit
def reduction_3(g_idata, g_odata):
    sdata = numba.cuda.shared.array(shape=THREADS, dtype=numba.int32)
    i = numba.cuda.grid(1)
    sdata[numba.cuda.threadIdx.x] = g_idata[i]
    numba.cuda.syncthreads()
    #do reduction in shared memory
    s = numba.cuda.blockDim.x // 2
    while s > 0:
        if numba.cuda.threadIdx.x < s:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + s]
        numba.cuda.syncthreads()
        s >>= 1
    numba.cuda.syncthreads()
    # write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = sdata[0]
    #implement second reduction for the summed array
    numba.cuda.syncthreads()
    s = numba.cuda.blockDim.x // 2
    while s > 0:
        if numba.cuda.threadIdx.x < s:
            g_odata[numba.cuda.threadIdx.x] += g_odata[numba.cuda.threadIdx.x + s]
        numba.cuda.syncthreads()
        s >>= 1
    numba.cuda.syncthreads()
    #write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = g_odata[0];
    numba.cuda.syncthreads()
