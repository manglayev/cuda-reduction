from helpers import *
from reduction_1 import *
from reduction_2 import *
from reduction_3 import *
from reduction_4 import *
from reduction_5 import *
from reduction_6 import *
from reduction_7 import *

import timeit

def callReduction(a, b):
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)

    if VARIANT == 1:
        reduction_1[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 2:
        reduction_2[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 3:
        reduction_3[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 4:
        reduction_4[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 5:
        reduction_5[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 6:
        reduction_6[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 7:
        reduction_7[BLOCKS, THREADS](dev_a, dev_b)
    sum = dev_b.copy_to_host()
    return sum

if __name__ == "__main__":
    print("VARIANT: ", VARIANT)
    a = np.ones(BLOCKS*THREADS, dtype=np.int32)
    b = np.ones(1, dtype=np.int32)
    start = timeit.default_timer()
    b = callReduction(a, b)
    print("GPU RESULTS: VARIANT =", VARIANT, "; b = ",b[0],"; elapsed time: ",str(round(timeit.default_timer() - start, 5)),"ms")
    sum = np.sum(a)
    print("CPU RESULTS:", sum)
