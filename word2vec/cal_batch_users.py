# -*- coding: utf-8 -*-
from datetime import datetime
from joblib import Parallel, delayed
from calculate_matrix import calculate_matrix
from functools import reduce
import gc

ITEM_NUM = 4318201

def cal_batch(user_logs, user_times_map):
    user_log_times = []
    for user_id, user_log in user_logs:
        user_log_times.append((user_log, user_times_map[user_id]))
    m - len(user_log_times)
    mat = lil_matrix((ITEM_NUM+1, ITEM_NUM+1), dtype=np.float32)
    for i, idx in enumerate(range(0, m, 1000 * 4)):
        batch_logs = user_log_times[idx: idx+1000*4]
        print('batch start')
        res = Parallel(n_jobs=4)(delayed(calculate_matrix)(batch_logs[start: start+1000],) for start in range(0, len(batch_logs), 1000))
        res = reduce(lambda a, b: a + b, res)
        print('The %d' % i + ' batch users are finished.')
        print(datetime.datetime.now().strftime('%H:%M:%S'))
        del res
        gc.collect()