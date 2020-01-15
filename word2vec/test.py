from calculate_matrix import calculate_matrix
from time import perf_counter
from scipy.sparse import lil_matrix, csr_matrix
import numpy as np
from joblib import Parallel, delayed
from functools import reduce

ITEM_NUM = 100000
item_log = list(range(0, int(1e2)))
s = perf_counter()
print('batch start')
res = Parallel(n_jobs=4)(delayed(calculate_matrix)(item_log,) for i in range(100))
res = reduce(lambda a, b: a + b, res)
# for i in range(100):
#     mat = calculate_matrix(item_log)
    # if i == 0:
    #     mat = csr_matrix((values, (rows, values)), shape=(ITEM_NUM, ITEM_NUM), dtype=np.float32).tolil()
    # else:
    #     mat += csr_matrix((values, (rows, values)), shape=(ITEM_NUM, ITEM_NUM), dtype=np.float32).tolil()
# res = csr_matrix((values, (rows, values)), shape=(ITEM_NUM, ITEM_NUM), dtype=np.float32).tolil()
print('batch finish')
print("cost time is %s s" % (perf_counter() - s))
# print(rows)