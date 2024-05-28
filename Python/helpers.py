import numpy as np
import numba

from numba import cuda

BLOCKS = 1024
THREADS = 1024
CUDASIZE = BLOCKS * THREADS
VARIANT = 2
