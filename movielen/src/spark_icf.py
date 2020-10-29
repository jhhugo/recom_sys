# -*- coding: utf-8 -*-
import findspark
import gc
from joblib import dump, load
import numpy as np   
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, LongType, StringType, IntegerType
import pyspark.sql.functions as F
from pyspark import SparkContext, SparkConf
from tqdm import tqdm
from itertools import permutations
from collections import defaultdict
import time

spark = SparkSession.builder.appName("item cf on spark").master("local[8]").getOrCreate()

sc = spark.sparkContext

schema = StructType([StructField('userId', IntegerType(), True), 
                     StructField('movieId', IntegerType(), True), 
                     StructField('rating', LongType(), True), 
                     StructField('timestamp', IntegerType(), True)])
ratings = spark.read.csv(r'D:\Users\hao.guo\deepctr\recsys\movielen\ml-20m\ratings_small.csv', header=True)

ratings = ratings.withColumn('rating', ratings['rating'].cast('int'))
ratings_rdd = ratings.select(['userId', 'movieId', 'rating']).rdd
# ratings_rdd = ratings_rdd.sample(withReplacement=False, fraction=0.5, seed=2020)
train_rdd, test_rdd = ratings_rdd.randomSplit([0.7, 0.3], seed=2020)
# train_rdd = train_rdd.cache()
# test_rdd = test_rdd.cache()

print('item cf start......')
s = time.perf_counter()

createCombiner = lambda v: [v]
mergeValue = lambda agg, v: agg + [v]
mergeCombiners = lambda agg1, agg2: agg1 + agg2

# 代替groupbykey
train_user_items = train_rdd.map(lambda s: ('user_' + s['userId'], [('item_' + s['movieId'], s['rating'])])).reduceByKey(lambda p1, p2: p1 + p2)
train_item_norm_dict = train_rdd.map(lambda s: ('item_' + s['movieId'], s['rating'] ** 2)).reduceByKey(lambda p1, p2: p1 + p2).mapValues(lambda v: np.sqrt(v)).collectAsMap()

train_item_norm_dict = sc.broadcast(train_item_norm_dict)

'''
        获得2个物品所有的user-user对得分组合:
        (item1_id,item2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
'''
def findpairs(pairs):
    res = []
    # 考虑活跃用户的影响, 冷门的物品兴趣才更加有把握相似
    m = len(pairs)
    for u1, u2 in permutations(pairs, 2):
        res.append(((u1[0], u2[0]), [(u1[1] / np.log1p(m), u2[1] / np.log1p(m))]))
        # res.append(((u1[0], u2[0]), (u1[1], u2[1])))
    return res

pairwise_items = train_user_items.filter(lambda p: len(p[1]) > 1).map(lambda p: p[1]).flatMap(lambda p: findpairs(p)).reduceByKey(lambda p1, p2: p1 + p2)

'''
    计算余弦相似度，找到最近的N个邻居:
    (item1,item2) ->    (similarity,co_raters_count)
'''

def cosine(product, r1_norm, r2_norm):
    fenmu = r1_norm * r2_norm
    return product / fenmu if fenmu else 0.0

def calcSim(pairs, item_norm_dict):
    ''' 
        对每个item对，根据打分计算余弦距离，并返回共同打分的user个数
    '''
    # 其他位置为0
    sum_xy, n = 0.0, 0

    for rating_pair in pairs[1]:
        sum_xy += rating_pair[0] * rating_pair[1]
        n += 1

    cos_sim = cosine(sum_xy, item_norm_dict[pairs[0][0]], item_norm_dict[pairs[0][1]])
    return pairs[0], (cos_sim, n)

def keyOnFirstItem(pairs):
    '''
        对于每个item-item对，用第一个item做key
    '''
    (item1_id, item2_id) = pairs[0]
    return item1_id,(item2_id, pairs[1])

def nearestNeighbors(item_id, co_pairs, n):
    '''
        选出相似度最高的N个物品
    '''
    # 归一化
    max_w = np.max([pairs[0] for i, pairs in co_pairs])
    scored_pairs = [(i, (pairs[0] / max_w if max_w != 0 else 0.0, pairs[1])) for i, pairs in co_pairs]
    scored_pairs.sort(key=lambda x: x[1][0], reverse=True)
    return item_id, scored_pairs[:n]

def item_sim(pairwise_items, train_item_norm_dict, n):
    item_sims = pairwise_items.map(lambda p: calcSim(p, train_item_norm_dict.value)).map(keyOnFirstItem).combineByKey(createCombiner, mergeValue, mergeCombiners).map(lambda p: nearestNeighbors(p[0], p[1], n))
    return item_sims


'''
    为每个用户计算Top N的推荐
    user_id -> [item1,item2,item3,...]
'''
def topNRecommendations(user_id, items_with_rating, item_sim_w, n):
    '''
        根据最近的N个物品进行推荐
    '''

    totals = defaultdict(int)

    user_items = [i for i, r in items_with_rating]

    for item, rating in items_with_rating:
        # 遍历邻居的打分
        nearest_neighbors = item_sim_w.get(item, None)

        if nearest_neighbors:
            for neighbor, (sim,count) in nearest_neighbors:
                if neighbor in user_items:
                    continue
                # 更新推荐度和相近度
                totals[neighbor] += sim * rating

    # 按照推荐度降序排列
    totals = sorted(totals.items(), key=lambda p: p[1], reverse=True)

    # 推荐度的item
    ranked_items = [x[0] for x in totals]

    return user_id, ranked_items[:n]

def Precision(user_id, pui, test, n):
    hit = 0
    all = 0
    tui = test[user_id]
    for item in pui:
        if item in tui:
            hit += 1
    all += n
    return (hit, all)

def Recall(user_id, pui, test):
    hit = 0
    all = 0
    tui = test[user_id]
    for item in pui:
        if item in tui:
            hit += 1
    all += len(tui)
    return (hit, all)

def Coverage(user_id, pui, train_items, test):
    recommend_items = set()
    all_items = set()
    tui = test[user_id]
    for item, rating in train_items[user_id]:
        all_items.add(item)
    for item in pui:
        recommend_items.add(item)
    return recommend_items, all_items

def Popularity(user_id, pui, item_popularity):
    ret = 0
    n = 0
    for item in pui:
        ret += np.log1p(item_popularity[item])
        n += 1
    return ret, n

def popularity(pairs):
    popular = set()
    for user, rating in pairs[1]:
        popular.add(user)
    return pairs[0], len(popular)

train_item_users = train_rdd.map(lambda s: ('item_' + s['movieId'], [('user_' + s['userId'], s['rating'])])).reduceByKey(lambda p1, p2: p1 + p2)
# item流行度字典
item_popularity = sc.broadcast(train_item_users.map(popularity).collectAsMap())

user_item = test_rdd.map(lambda s: ('user_' + s['userId'], ['item_' + s['movieId']])).reduceByKey(lambda p1, p2: p1 + p2)
user_item_hist = user_item.collectAsMap()
test_users = list(user_item_hist.keys())
test_users_items = sc.broadcast(user_item_hist)
train_items = sc.broadcast(train_user_items.collectAsMap())

def eval(test_users, test_users_items, item_sims_w, train_items, item_popularity, n):
# def eval(item_sims, train_user_items, item_popularity, test, n):
    # user_item = test.map(lambda s: ('user_' + s['userId'], ['item_' + s['movieId']])).reduceByKey(lambda p1, p2: p1 + p2)
    # user_item_hist = user_item.collectAsMap()
    # test_users = list(user_item_hist.keys())
    # test_users_items = sc.broadcast(user_item_hist)

    # item_sims_w = sc.broadcast(item_sims.collectAsMap())

    '''
        为每个用户计算Top N的推荐
        user_id -> [item1,item2,item3,...]
    '''
    # choose test recs, 有过历史浏览的用户
    user_item_recs = train_user_items.filter(lambda p: p[0] in test_users).map(
        lambda p: topNRecommendations(p[0], p[1], item_sims_w.value, n)).cache()
    
    # precision
    hit, all = user_item_recs.map(lambda p: Precision(p[0], p[1], test_users_items.value, n)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    p = hit / (all * 1.0)
    # recall
    hit, all = user_item_recs.map(lambda p: Recall(p[0], p[1], test_users_items.value)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    r = hit / (all * 1.0)
    # Coverage
    # train_items = sc.broadcast(train_user_items.collectAsMap())
    recommend_items, all_items = user_item_recs.map(lambda p: Coverage(p[0], p[1], train_items.value, test_users_items.value)).reduce(lambda x, y: (x[0].union(y[0]), x[1].union(y[1])))
    c = len(recommend_items) / (len(all_items) * 1.0)
    # popularity
    recom_popularity, all = user_item_recs.map(lambda p: Popularity(p[0], p[1], item_popularity.value)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    popularity = recom_popularity / (all * 1.0)
    # del user_item_recs, user_item_hist, test_users, test_users_items, item_popularity, item_sims_w, train_items
    del user_item_recs, test_users, test_users_items, item_popularity, item_sims_w, train_items
    gc.collect()
    return p, r, c, popularity

# 生成训练集各用户的相似矩阵
for k in [5, 10, 20, 40, 80, 160]:
    # 物品最相近的N个物品
    item_sims = item_sim(pairwise_items, train_item_norm_dict, k)
    item_sims_w = sc.broadcast(item_sims.collectAsMap())
    print('eval model start......')
    # p, r, c, popularity = eval(item_sims_w, train_user_items, item_popularity, test_rdd, 50)
    p, r, c, popularity = eval(test_users, test_users_items, item_sims_w, train_items, item_popularity, 50)
    print('top %s model: %s, %s, %s, %s' % (k, p, r, c, popularity))
    del item_sims, item_sims_w
    gc.collect()

# user_sims = user_sim(pairwise_users, train_user_norm_dict, 10)
# p10, r10, c10, popularity10 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
# print('top 10 model: %s, %s, %s, %s' % (p10, r10, c10, popularity10))
# del user_sims
# gc.collect()

# user_sims = user_sim(pairwise_users, train_user_norm_dict, 20)
# p20, r20, c20, popularity20 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
# print('top 20 model: %s, %s, %s, %s' % (p20, r20, c20, popularity20))
# del user_sims
# gc.collect()

# user_sims = user_sim(pairwise_users, train_user_norm_dict, 40)
# p40, r40, c40, popularity40 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
# print('top 40 model: %s, %s, %s, %s' % (p40, r40, c40, popularity40 ))
# del user_sims
# gc.collect()

# user_sims = user_sim(pairwise_users, train_user_norm_dict, 80)
# p80, r80, c80, popularity80 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
# print('top 80 model: %s, %s, %s, %s' % (p80, r80, c80, popularity80))
# del user_sims
# gc.collect()
print('cost time %s min' % ((time.perf_counter() - s) / 60))