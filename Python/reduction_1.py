from numba import cuda, int32

@cuda.jit
def reduction_1(g_idata, g_odata):
    sdata = cuda.shared.array(shape=THREADS, dtype=int32)
    i = cuda.grid(1)
    sdata[cuda.threadIdx.x] = g_idata[i]
    cuda.syncthreads()
    #do reduction in shared mem
    s = 1
    for s in range(1, cuda.blockDim.x, s*2):
        if cuda.threadIdx.x % (2*s) == 0:
          sdata[cuda.threadIdx.x] += sdata[cuda.threadIdx.x + s]
        cuda.syncthreads()
    # write result for this block to global mem
    if cuda.threadIdx.x == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]
    #implement second reduction for the summed array
    s = 1
    for s in range(1, cuda.blockDim.x, s*2):
        if cuda.threadIdx.x % (2*s) == 0:
            g_odata[cuda.threadIdx.x] += g_odata[cuda.threadIdx.x + s];
        cuda.syncthreads();
    #write result for this block to global mem
    if cuda.threadIdx.x == 0:
        g_odata[cuda.blockIdx.x] = g_odata[0];
