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

spark = SparkSession.builder.appName("user cf on spark").master("local[8]").getOrCreate()

sc = spark.sparkContext

schema = StructType([StructField('userId', IntegerType(), True), 
            StructField('movieId', IntegerType(), True), 
            StructField('rating', LongType(), True), 
            StructField('timestamp', IntegerType(), True)])
ratings = spark.read.csv(r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\ml-20m\ratings_small.csv', header=True)

ratings = ratings.withColumn('rating', ratings['rating'].cast('int'))
ratings_rdd = ratings.select(['userId', 'movieId', 'rating']).rdd
# ratings_rdd = ratings_rdd.sample(withReplacement=False, fraction=0.5, seed=2020)
train_rdd, test_rdd = ratings_rdd.randomSplit([0.7, 0.3], seed=2020)
train_rdd = train_rdd.cache()
test_rdd = test_rdd.cache()

print('user cf start......')
s = time.perf_counter()

createCombiner = lambda v: [v]
mergeValue = lambda agg, v: agg + [v]
mergeCombiners = lambda agg1, agg2: agg1 + agg2
train_item_users = train_rdd.map(lambda s: ('item_' + s['movieId'], ('user_' + s['userId'], s['rating']))).combineByKey(createCombiner, mergeValue, mergeCombiners).cache()
train_user_norm_dict = train_rdd.map(lambda s: ('user_' + s['userId'], s['rating'] ** 2)).reduceByKey(lambda p1, p2: p1 + p2).mapValues(lambda v: np.sqrt(v)).collectAsMap()

train_user_norm_dict = sc.broadcast(train_user_norm_dict)

'''
        获得2个用户所有的item-item对得分组合:
        (user1_id,user2_id) -> [(rating1,rating2),
                                (rating1,rating2),
                                (rating1,rating2),
                                ...]
'''
def findpairs(pairs):
    res = []
    for u1, u2 in permutations(pairs, 2):
        res.append(((u1[0], u2[0]), (u1[1], u2[1])))
    return res
pairwise_users = train_item_users.filter(lambda p: len(p[1]) > 1).map(lambda p: p[1]).flatMap(lambda p: findpairs(p)).combineByKey(createCombiner, mergeValue, mergeCombiners).cache()

'''
    计算余弦相似度，找到最近的N个邻居:
    (user1,user2) ->    (similarity,co_raters_count)
'''

def cosine(product, r1_norm, r2_norm):
    fenmu = r1_norm * r2_norm
    return product / fenmu if fenmu else 0.0

def calcSim(pairs, user_norm_dict):
    ''' 
        对每个user对，根据打分计算余弦距离，并返回共同打分的item个数
    '''
    # 其他位置为0
    sum_xy, n = 0.0, 0
    
    for rating_pair in pairs[1]:
        sum_xy += rating_pair[0] * rating_pair[1]
        n += 1

    cos_sim = cosine(sum_xy, user_norm_dict[pairs[0][0]], user_norm_dict[pairs[0][1]])
    return pairs[0], (cos_sim, n)

def keyOnFirstUser(pairs):
    '''
        对于每个user-user对，用第一个user做key(好像有点粗暴...)
    '''
    (user1_id, user2_id) = pairs[0]
    return user1_id,(user2_id, pairs[1])

def nearestNeighbors(pairs, n):
    '''
        选出相似度最高的N个邻居
    '''
    pairs[1].sort(key=lambda x: x[1][0], reverse=True)
    return pairs[0], pairs[1][:n]

def user_sim(pairwise_users, train_user_norm_dict, n):
    user_sims = pairwise_users.map(lambda p: calcSim(p, train_user_norm_dict.value)).map(keyOnFirstUser).combineByKey(createCombiner, mergeValue, mergeCombiners).map(lambda p: nearestNeighbors(p, n))
    return user_sims


''' 
    对每个用户的打分记录整理成如下形式
    user_id -> [(item_id_1, rating_1),
                [(item_id_2, rating_2),
                ...]
'''

train_user_items = sc.broadcast(train_rdd.map(lambda s: ('user_' + s['userId'], ('item_' + s['movieId'], s['rating']))).combineByKey(createCombiner, mergeValue, mergeCombiners).collectAsMap())

'''
    为每个用户计算Top N的推荐
    user_id -> [item1,item2,item3,...]
'''
def topNRecommendations(user_id, user_sims, users_with_rating, n):
    '''
        根据最近的N个邻居进行推荐
    '''

    totals = defaultdict(int)
    sim_sums = defaultdict(int)

    for ind, (neighbor,(sim,count)) in enumerate(user_sims):

        # 遍历邻居的打分
        unscored_items = users_with_rating.get(neighbor,None)

        if unscored_items:
            for (item,rating) in unscored_items:
                if item not in users_with_rating.get(user_id, []):

                # 更新推荐度和相近度
                    totals[item] += sim * rating
                    sim_sums[item] += sim

    # 归一化
    scored_items = [(total/(1e-10 if sim_sums[item] == 0 else sim_sums[item]), item) for item,total in totals.items()]

    # 按照推荐度降序排列
    scored_items.sort(reverse=True)

    # 推荐度的item
    ranked_items = [x[1] for x in scored_items]

    return user_id, ranked_items[:n]

def Precision(user_id, pui, test):
    hit = 0
    all = 0
    tui = test[user_id]
    for item in pui:
        if item in tui:
            hit += 1
    all += len(pui)
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
    return len(recommend_items), len(all_items)

def Popularity(user_id, pui, item_popularity):
    ret = 0
    n = 0
    for item in pui:
        ret += np.log1p(item_popularity[item])
        n += 1
    return ret, n

def popularity(pairs):
    popular = set()
    for item, rating in pairs[1]:
        popular.add(item)
    return pairs[0], len(popular)


item_popularity = sc.broadcast(train_item_users.map(popularity).collectAsMap())

def eval(user_sims, item_popularity, train_user_items, test, n):
    user_item = test.map(lambda s: ('user_' + s['userId'], 'item_' + s['movieId'])).combineByKey(createCombiner, mergeValue, mergeCombiners)
    user_item_hist = user_item.collectAsMap()
    test_users = list(user_item_hist.keys())
    test_users_items = sc.broadcast(user_item_hist)

    '''
        为每个用户计算Top N的推荐
        user_id -> [item1,item2,item3,...]
    '''
    # choose test recs
    user_item_recs = user_sims.filter(lambda p: p[0] in test_users).map(
        lambda p: topNRecommendations(p[0], p[1], train_user_items.value, n)).cache()
    
    # precision
    hit, all = user_item_recs.map(lambda p: Precision(p[0], p[1], test_users_items.value)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    p = hit / (all * 1.0)
    # recall
    hit, all = user_item_recs.map(lambda p: Recall(p[0], p[1], test_users_items.value)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    r = hit / (all * 1.0)
    # Coverage
    recommend_items, all_items = user_item_recs.map(lambda p: Coverage(p[0], p[1], train_user_items.value, test_users_items.value)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    c = recommend_items / (all_items * 1.0)
    # popularity
    recom_popularity, all = user_item_recs.map(lambda p: Popularity(p[0], p[1], item_popularity.value)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
    popularity = recom_popularity / (all * 1.0)
    del user_item_recs
    gc.collect()
    return p, r, c, popularity

# 生成训练集各用户的相似矩阵
user_sims = user_sim(pairwise_users, train_user_norm_dict, 5)
print('eval model start......')
p5, r5, c5, popularity5 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
print('top 5 model: %s, %s, %s, %s' % (p5, r5, c5, popularity5))
del user_sims
gc.collect()

user_sims = user_sim(pairwise_users, train_user_norm_dict, 10)
p10, r10, c10, popularity10 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
print('top 10 model: %s, %s, %s, %s' % (p10, r10, c10, popularity10))
del user_sims
gc.collect()

user_sims = user_sim(pairwise_users, train_user_norm_dict, 20)
p20, r20, c20, popularity20 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
print('top 20 model: %s, %s, %s, %s' % (p20, r20, c20, popularity20))
del user_sims
gc.collect()

user_sims = user_sim(pairwise_users, train_user_norm_dict, 40)
p40, r40, c40, popularity40 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
print('top 40 model: %s, %s, %s, %s' % (p40, r40, c40, popularity40 ))
del user_sims
gc.collect()

user_sims = user_sim(pairwise_users, train_user_norm_dict, 80)
p80, r80, c80, popularity80 = eval(user_sims, item_popularity, train_user_items, test_rdd, 100)
print('top 80 model: %s, %s, %s, %s' % (p80, r80, c80, popularity80))
del user_sims
gc.collect()
print('cost time %s min' % ((time.perf_counter() - s) / 60))