# -*- coding: utf-8 -*-
import pandas as pd
import random
import math
data = pd.read_csv(r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\ml-20m\ratings.csv')
data = data[['userId', 'movieId']].values

def SplitData(data, M, k, seed):
    test = []
    train = []
    random.seed(seed)
    for user, item in data:
        if random.randint(0,M) == k:
            test.append([user,item])
        else:
            train.append([user,item])
    return train, test

def Recommend(user, train, W, K):
    rank = defaultdict(int)
    interacted_items = list(map(train[user], lambda p: p[0]))
    for v, wuv in sorted(W[user].items, key=lambda p: p[1], reverse=True)[0:K]:
        for i, rvi in train[v].items:
            if i in interacted_items:
            #we should filter items user interacted before
                continue
        rank[i] += wuv * rvi
    return rank

def Recall(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = Recommend(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(tu)
    return hit / (all * 1.0)

def Precision(train, test, N):
    hit = 0
    all = 0
    for user in train.keys():
        tu = test[user]
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += N
    return hit / (all * 1.0)

def Coverage(train, test, N):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)

def Popularity(train, test, N):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = GetRecommendation(user, N)
        for item, pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret

import numpy as np
from collections import defaultdict

def UserSimilarity(train):
    # build inverse table for item_users
    item_users = dict()
    users = set()
    for u, items in train.items():
        for i in items.keys():
            # if i not in item_users:
            #     item_users[i] = set()
            item_users.setdefault(i, set()).add(u)
        users.add(u)
    #calculate co-rated items between users
    # C = dict()
    users_dict = {u: 0 for u in users}
    C = defaultdict(lambda : users_dict)
    # C = np.zeros((len(item_users), len(item_users)))
    # user items list
    N = defaultdict(int)
    for i, users in item_users.items():
        for u in users:
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C[u][v] += 1
    #calculate finial similarity matrix W
    W = dict()
    for u, related_users in C.items():
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W