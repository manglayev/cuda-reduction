from helpers import *

@numba.cuda.jit
def reduction_warp(sdata, tid):
    if VARIANT == 5:
        sdata[tid] += sdata[tid + 32]
        sdata[tid] += sdata[tid + 16]
        sdata[tid] += sdata[tid + 8]
        sdata[tid] += sdata[tid + 4]
        sdata[tid] += sdata[tid + 2]
        sdata[tid] += sdata[tid + 1]
    if (VARIANT == 6 or VARIANT == 7) and sdata.shape[0] == THREADS:
        if THREADS >= 64:
            sdata[tid] += sdata[tid + 32]
        if THREADS >= 32:
            sdata[tid] += sdata[tid + 16]
        if THREADS >= 16:
            sdata[tid] += sdata[tid + 8]
        if THREADS >= 8:
            sdata[tid] += sdata[tid + 4]
        if THREADS >= 4:
            sdata[tid] += sdata[tid + 2]
        if THREADS >= 2:
            sdata[tid] += sdata[tid + 1]
    if (VARIANT == 6 or VARIANT == 7) and sdata.shape[0] == BLOCKS_TO_FOUR:
        if BLOCKS_TO_FOUR >= 64:
            sdata[tid] += sdata[tid + 32]
        if BLOCKS_TO_FOUR >= 32:
            sdata[tid] += sdata[tid + 16]
        if BLOCKS_TO_FOUR >= 16:
            sdata[tid] += sdata[tid + 8]
        if BLOCKS_TO_FOUR >= 8:
            sdata[tid] += sdata[tid + 4]
        if BLOCKS_TO_FOUR >= 4:
            sdata[tid] += sdata[tid + 2]
        if BLOCKS_TO_FOUR >= 2:
            sdata[tid] += sdata[tid + 1]
