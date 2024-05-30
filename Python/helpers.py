import numpy as np
import numba

from numba import cuda

BLOCKS = 1024
THREADS = 1024
BLOCKS_TO_FOUR = int(BLOCKS/4)
CUDASIZE = BLOCKS * THREADS
VARIANT = 4
