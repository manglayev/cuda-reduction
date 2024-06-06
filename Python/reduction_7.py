from helpers import *
from reduction_warp import *

@numba.cuda.jit
def reduction_7(g_idata, g_odata):
    sdata = numba.cuda.shared.array(shape=THREADS, dtype=numba.int32)
    #each thread loads one element from global to shared mem
    i = numba.cuda.blockIdx.x * (THREADS*2) + numba.cuda.threadIdx.x
    gridSize = THREADS*2*numba.cuda.gridDim.x
    sdata[numba.cuda.threadIdx.x] = 0
    while i < CUDASIZE:
        sdata[numba.cuda.threadIdx.x] += g_idata[i] + g_idata[i+THREADS]
        i += gridSize
    numba.cuda.syncthreads()
    #do reduction in shared memory
    if THREADS >= 512:
        if numba.cuda.threadIdx.x < 256:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 256]
        numba.cuda.syncthreads();
    if THREADS >= 256:
        if numba.cuda.threadIdx.x < 128:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 128]
        numba.cuda.syncthreads()
    if THREADS >= 128:
        if numba.cuda.threadIdx.x < 64:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 64]
        numba.cuda.syncthreads();
    if numba.cuda.threadIdx.x < 32:
        reduction_warp(sdata, numba.cuda.threadIdx.x)
    numba.cuda.syncthreads()
    # write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = sdata[0]
    numba.cuda.syncthreads()
    #implement second reduction for the summed array
    i = numba.cuda.blockIdx.x * (THREADS*2) + numba.cuda.threadIdx.x
    gridSize = THREADS*2*numba.cuda.gridDim.x
    sdata[numba.cuda.threadIdx.x] = 0
    while i < CUDASIZE:
        sdata[numba.cuda.threadIdx.x] += g_odata[i] + g_odata[i+THREADS]
        i += gridSize
    numba.cuda.syncthreads()
    if THREADS >= 512:
        if numba.cuda.threadIdx.x < 256:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 256]
        numba.cuda.syncthreads();
    if THREADS >= 256:
        if numba.cuda.threadIdx.x < 128:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 128]
        numba.cuda.syncthreads()
    if THREADS >= 128:
        if numba.cuda.threadIdx.x < 64:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 64]
        numba.cuda.syncthreads()
    if numba.cuda.threadIdx.x < 32:
        reduction_warp(sdata, numba.cuda.threadIdx.x)
    numba.cuda.syncthreads();
    # write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = sdata[0]
    numba.cuda.syncthreads();

@numba.cuda.jit
def reduction_71(g_idata, g_odata):
    sdata = numba.cuda.shared.array(shape=THREADS, dtype=numba.int32)
    #each thread loads one element from global to shared mem
    i = numba.cuda.blockIdx.x * (THREADS*2) + numba.cuda.threadIdx.x
    gridSize = THREADS*2*numba.cuda.gridDim.x
    sdata[numba.cuda.threadIdx.x] = 0
    while i < CUDASIZE:
        sdata[numba.cuda.threadIdx.x] += g_idata[i] + g_idata[i+THREADS]
        i += gridSize
    numba.cuda.syncthreads()
    #do reduction in shared memory
    if THREADS >= 1024:
        if numba.cuda.threadIdx.x < 512:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 512]
        numba.cuda.syncthreads();
    if THREADS >= 512:
        if numba.cuda.threadIdx.x < 256:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 256]
        numba.cuda.syncthreads();
    if THREADS >= 256:
        if numba.cuda.threadIdx.x < 128:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 128]
        numba.cuda.syncthreads()
    if THREADS >= 128:
        if numba.cuda.threadIdx.x < 64:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 64]
        numba.cuda.syncthreads();
    if numba.cuda.threadIdx.x < 32:
        reduction_warp(sdata, numba.cuda.threadIdx.x)
    numba.cuda.syncthreads()
    # write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = sdata[0]
    numba.cuda.syncthreads()

@numba.cuda.jit
def reduction_72(g_idata, g_odata):
    sdata = numba.cuda.shared.array(shape=BLOCKS_TO_FOUR, dtype=numba.int32)
    #each thread loads one element from global to shared mem
    i = numba.cuda.blockIdx.x * (BLOCKS_TO_FOUR*2) + numba.cuda.threadIdx.x
    gridSize = THREADS*2*numba.cuda.gridDim.x
    sdata[numba.cuda.threadIdx.x] = 0
    while i < BLOCKS_TO_FOUR:
        sdata[numba.cuda.threadIdx.x] += g_idata[i] + g_idata[i+BLOCKS_TO_FOUR]
        i += gridSize
    numba.cuda.syncthreads()
    #do reduction in shared memory
    if BLOCKS_TO_FOUR >= 512:
        if numba.cuda.threadIdx.x < 256:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 256]
        numba.cuda.syncthreads();
    if BLOCKS_TO_FOUR >= 256:
        if numba.cuda.threadIdx.x < 128:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 128]
        numba.cuda.syncthreads()
    if BLOCKS_TO_FOUR >= 128:
        if numba.cuda.threadIdx.x < 64:
            sdata[numba.cuda.threadIdx.x] += sdata[numba.cuda.threadIdx.x + 64]
        numba.cuda.syncthreads();
    if numba.cuda.threadIdx.x < 32:
        reduction_warp(sdata, numba.cuda.threadIdx.x)
    numba.cuda.syncthreads()
    # write result for this block to global memory
    if numba.cuda.threadIdx.x == 0:
        g_odata[numba.cuda.blockIdx.x] = sdata[0]
    numba.cuda.syncthreads()
