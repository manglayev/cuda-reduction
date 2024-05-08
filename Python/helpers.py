import numpy as np
import numba

from numba import cuda

BLOCKS = 8
THREADS = 512
CUDASIZE = BLOCKS * THREADS
VARIANT = 7
