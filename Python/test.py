from numba.cuda.cudadrv import enums
from numba import cuda

def test():
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
    #print("Total global memory:", getattr(device, "TOTAL_GLOBAL_MEMORY"))
    print("Total constant memory:", getattr(device, "TOTAL_CONSTANT_MEMORY"))
    print("Multiprocessor count:", getattr(device, "MULTIPROCESSOR_COUNT"))
    print("Shared memory per block:", getattr(device, "MAX_SHARED_MEMORY_PER_BLOCK"))
    print("Registers per block:", getattr(device, "MAX_REGISTERS_PER_BLOCK"))
    print("Threads in warp:", getattr(device, "WARP_SIZE"))
    print("Max threads Per Block:", getattr(device, "MAX_THREADS_PER_BLOCK"))
    print("Max thread dimensions:(", getattr(device, "MAX_BLOCK_DIM_X"), getattr(device, "MAX_BLOCK_DIM_Y"), getattr(device, "MAX_BLOCK_DIM_Z"),")");
    print("Max grid dimensions:(", getattr(device, "MAX_GRID_DIM_X"), getattr(device, "MAX_GRID_DIM_Y"), getattr(device, "MAX_GRID_DIM_Z"),")");
    print("  --- General information for device END ---")
test()
