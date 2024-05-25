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
        #reduction_1[BLOCKS, THREADS](dev_a, dev_b)
        reduction_10[BLOCKS, THREADS](dev_a, dev_b)
        reduction_10[1, BLOCKS](dev_b, dev_b)
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
    device = cuda.get_current_device()
    #attributes = [name.replace("CU_DEVICE_ATTRIBUTE_", "") for name in dir(enums) if name.startswith("CU_DEVICE_ATTRIBUTE_")]
    #for attribute in attributes:
    #    print(attribute, '=', getattr(device, attribute))
    #print("Name:",cuda.cudadrv.driver.Device(0).name)
    #print("Compute capability:",cuda.cudadrv.driver.Device(0).compute_capability)
    print("  --- General information for device START ---");
    print("Name:",cuda.cudadrv.driver.Device(0).name)
    print("Compute capability:",cuda.cudadrv.driver.Device(0).compute_capability)
    print("Clock rate:", getattr(device, "CLOCK_RATE"))
    print("Total global memory:", numba.cuda.cudadrv.devices.get_context(0).get_memory_info()[1])
    print("Total constant memory:", getattr(device, "TOTAL_CONSTANT_MEMORY"))
    print("Multiprocessor count:", getattr(device, "MULTIPROCESSOR_COUNT"))
    print("Shared memory per block:", getattr(device, "MAX_SHARED_MEMORY_PER_BLOCK"))
    print("Registers per block:", getattr(device, "MAX_REGISTERS_PER_BLOCK"))
    print("Threads in warp:", getattr(device, "WARP_SIZE"))
    print("Max threads Per Block:", getattr(device, "MAX_THREADS_PER_BLOCK"))
    print("Max thread dimensions:(", getattr(device, "MAX_BLOCK_DIM_X"), getattr(device, "MAX_BLOCK_DIM_Y"), getattr(device, "MAX_BLOCK_DIM_Z"),")");
    print("Max grid dimensions:(", getattr(device, "MAX_GRID_DIM_X"), getattr(device, "MAX_GRID_DIM_Y"), getattr(device, "MAX_GRID_DIM_Z"),")");
    print("  --- General information for device END ---")

    a = np.ones(BLOCKS*THREADS, dtype=np.int32)
    b = np.ones(1, dtype=np.int32)
    start = timeit.default_timer()
    b = callReduction(a, b)
    print("GPU RESULTS: VARIANT =", VARIANT, "; b = ",b[0],"; elapsed time: ",str(round(timeit.default_timer() - start, 5)),"ms")
    sum = np.sum(a)
    print("CPU RESULTS:", sum)
