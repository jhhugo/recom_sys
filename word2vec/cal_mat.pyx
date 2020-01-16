from scipy.sparse import lil_matrix
from libc.math cimport log1p
import numpy as np
cdef int ITEM_NUM = 4318201
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef calculate_matrix(list user_log_times):
    cdef int i1, i2, idx, j
    mat = lil_matrix((ITEM_NUM+1, ITEM_NUM+1), dtype=np.float32)
    for user_log, user_time in user_log_times:
        for idx, i1 in enumerate(user_log):
            for i2 in user_log[(idx+1):]:
                weight = 1 / (log1p(user_time))
                mat[i1, i2] += weight
                mat[i2, i1] += weight 
    return mat