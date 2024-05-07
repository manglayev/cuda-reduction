from helpers import *

@numba.cuda.jit
def reduction_warp(sdata, tid):
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
