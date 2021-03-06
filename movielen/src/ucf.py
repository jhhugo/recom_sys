# -*- coding: utf-8 -*-
from collections import defaultdict
import time
import gc
import pandas as pd
import numpy as np

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
def UserSimilarity(item_users, users):
    # build inverse table for item_users
#     item_users = dict()
#     users = set()
#     for u, items in data.items():
#         for i in items.keys():
#             # if i not in item_users:
#             #     item_users[i] = set()
#             item_users.setdefault(i, set()).add(u)
#         users.add(u)
    #calculate co-rated items between users
    # C = dict()
    C = defaultdict(dict)
    # C = np.zeros((len(item_users), len(item_users)))
    # user items list
    N = defaultdict(int)
    print('共现矩阵计算')
    for i, pairs in tqdm(item_users.items()):
        for u, r1 in pairs:
            N[u] += r1 ** 2
            for v, r2 in pairs:
                if u == v:
                    continue
                if v not in list(C[u]):
                    C[u][v] = 0
                C[u][v] += (r1 * r2 / np.log1p(len(pairs)))
    #calculate finial similarity matrix W
    W = defaultdict(dict)
    print('相似矩阵计算')
    for u, related_users in tqdm(C.items()):
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W
s = time.perf_counter()
W = UserSimilarity(train_item_users, train_users)
print('cost time %s min' % ((time.perf_counter() - s) / 60))

def Recommend(user, train, W, K):
    rank = defaultdict(int)
    sim_sums = defaultdict(int)
    
    interacted_items = list(map(lambda p: p[0], train[user]))
    for v, wuv in sorted(W[user].items(), key=lambda p: p[1], reverse=True)[0:K]:
        for i, rvi in train[v]:
            if i in interacted_items:
            #we should filter items user interacted before
                continue
            rank[i] += wuv * rvi
            sim_sums[i] += wuv
    res = [(sim / 1, i) for i, sim in rank.items()]
    res.sort(reverse=True)
    ranked_items = [x[1] for x in res]
#     sorted(rank.items(), key=lambda p: p[1], reverse=True)
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

def eval(train_user_items, test_user_items, W, N):
    p = Precision(train_user_items, test_user_items, W, N)
    r = Recall(train_user_items, test_user_items, W, N)
    c = Coverage(train_user_items, test_user_items, W, N)
    pop = Popularity(train_user_items, test_user_items, W, N)
    return (p, r, c, pop)
    
# print(eval(train_user_items, test_user_items, W, 5))
print(eval(train_user_items, test_user_items, W, 10))
# print(eval(train_user_items, test_user_items, W, 20))
# print(eval(train_user_items, test_user_items, W, 30))
# print(eval(train_user_items, test_user_items, W, 40))
# print(eval(train_user_items, test_user_items, W, 50))