from helpers import *
from reduction_1 import *
from reduction_2 import *
from reduction_3 import *

def callReduction(a, b):
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)
    if VARIANT == 1:
        reduction_1[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 2:
        reduction_2[BLOCKS, THREADS](dev_a, dev_b)
    if VARIANT == 3:
        reduction_3[BLOCKS, THREADS](dev_a, dev_b)
    sum = dev_b.copy_to_host()
    return sum

if __name__ == "__main__":
    a = np.ones(BLOCKS*THREADS, dtype=np.int32)
    b = np.ones(1, dtype=np.int32)
    b = callReduction(a, b)
    print("GPU RESULTS: ", b[0])
    sum = np.sum(a)
    print("CPU RESULTS: ", sum)
