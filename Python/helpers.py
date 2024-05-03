import numpy as np
import numba

from numba import cuda

BLOCKS = 8
THREADS = 512
VARIANT = 2
