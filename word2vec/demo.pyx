from time import perf_counter
from libc.math cimport sqrt
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
# @cython.boundscheck(False) 和 @cython.wraparound(False) 两个修饰符用来关闭 Cython 的边界检查
cdef double cal(int num):
    cdef int i
    cdef double sum
    sum = 0.0
    for i in range(num):
        sum += sqrt(num)
    return sum

def main():
    s = perf_counter()
    res = cal(int(1e8))
    print('cost time is %s s' % (perf_counter() - s))

# main()