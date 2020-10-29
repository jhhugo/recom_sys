from collections import defaultdict
import time
import gc
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

s = time.perf_counter()
train = pd.read_csv(r'E:\ML_study\学习笔记\推荐\movielens\ml-20m\ratings_train.csv', engine='python')
test = pd.read_csv(r'E:\ML_study\学习笔记\推荐\movielens\ml-20m\ratings_test.csv', engine='python')

def combine(data):
    item_users = dict()
    user_items = dict()
    users = set()
    data['userId'] = 'user_' + data['userId'].astype(str)
    data['movieId'] = 'movie_' + data['movieId'].astype(str)
    for u, item, r in data[['userId', 'movieId', 'rating']].values:
        item_users.setdefault(item, []).append((u, r))
        user_items.setdefault(u, []).append((item, r))
        users.add(u)
    del data
    gc.collect()
    return item_users, user_items, users

train_item_users, train_user_items, train_users = combine(train)
test_item_users, test_user_items, test_users = combine(test)
print('cost time %s min' % ((time.perf_counter() - s) / 60))

import math
from tqdm import tqdm_notebook

def ItemSimilarity(user_items):
    # build inverse table for user_items
    C = defaultdict(dict)

    # item users list
    N = defaultdict(int)
    print('共现矩阵计算')
    for u, pairs in tqdm(user_items.items()):
        for i, r1 in pairs:
            N[i] += r1 ** 2
            for j, r2 in pairs:
                if i == j:
                    continue
                if j not in list(C[i]):
                    C[i][j] = 0
                C[i][j] += (r1 * r2 / np.log1p(len(pairs)))
    #calculate finial similarity matrix W
    W = defaultdict(dict)
    print('相似矩阵计算')
    for i, related_items in tqdm(C.items()):
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    # 归一化
    for k, v in W.items():
        max_w = np.max(list(v.values()))
        for j, w in v.items():
            v[j] = w / max_w
            
    return W
s = time.perf_counter()
W = ItemSimilarity(train_user_items)
print('cost time %s min' % ((time.perf_counter() - s) / 60))

def Recommend(user, train, W, K):
    rank = defaultdict(int)
    
    interacted_items = list(map(lambda p: p[0], train[user]))
    for item, rji in train[user]:
        for j, wij in sorted(W[item].items(), key=lambda p: p[1], reverse=True)[0:K]:
            if j in interacted_items:
            #we should filter items user interacted before
                continue
            rank[j] += wij * rji

    res = sorted(rank.items(), key=lambda p: p[1], reverse=True)
    ranked_items = [x[0] for x in res]
    return ranked_items[:50]

def Precision(train, test, W, N):
    hit = 0
    all = 0
    for user in test.keys():
        tu = list(map(lambda p: p[0], test[user]))
        rank = Recommend(user, train, W, N)
        for item in rank:
            if item in tu:
                hit += 1
        all += 50
    return hit / (all * 1.0)

def Recall(train, test, W, N):
    hit = 0
    all = 0
    for user in test.keys():
        tu = list(map(lambda p: p[0], test[user]))
        rank = Recommend(user, train, W, N)
        for item in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def Coverage(train, test, W, N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item, r in test[user]:
            all_items.add(item)
    for user in test.keys():
        rank = Recommend(user, train, W, N)
        for item in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

def Popularity(train, test, W, N):
    item_popularity = dict()
    for user, items in train.items():
        for item, r in items:
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in test.keys():
        rank = Recommend(user, train, W, N)
        for item in rank:
            ret += np.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret 

# map
def ap(rank_list, ground_truth):
    hit = 0
    sum_pres = 0
    for n, item in enumerate(rank_list, start=1):
        if item in ground_truth:
            hit += 1
            sum_pres += hit / n
    return sum_pres / hit

def map(recom_results,train_truth,njobs=1):
    p = Pool(njobs)
    res = []
    for user, rank_list in recom_results.items():
        res.append(p.apply_async(ap, args=(rank_list, train_truth[user])))
    p.close()
    p.join()
    aps = [r.get() for r in res]
    return np.sum(aps) / len(aps)

# ndcg(对于评分的)
def dcg(rank_list):
    sum_dcg = 0
    for n, (item, rating) in enumerate(rank_list.items(), start=1):
        sum_dcg += (2 ** (rating) - 1) / np.log2(n + 1)
    return sum_dcg

def ndcg(recom_results, njobs=1):
    p = Pool(njobs)
    res = []
    for user, rank_list in recom_results.items():
        res.append(p.apply_async(dcg, args=(rank_list,)))
    p.close()
    p.join()
    dcgs = [r.get() for r in res]
    return np.sum(dcgs) / len(dcgs)

def eval(train_user_items, test_user_items, W, N):
    p = Precision(train_user_items, test_user_items, W, N)
    r = Recall(train_user_items, test_user_items, W, N)
    c = Coverage(train_user_items, test_user_items, W, N)
    pop = Popularity(train_user_items, test_user_items, W, N)
    return (p, r, c, pop)

print(eval(train_user_items, test_user_items, W, 5))
print(eval(train_user_items, test_user_items, W, 10))
print(eval(train_user_items, test_user_items, W, 20))
print(eval(train_user_items, test_user_items, W, 30))
print(eval(train_user_items, test_user_items, W, 40))
print(eval(train_user_items, test_user_items, W, 50))