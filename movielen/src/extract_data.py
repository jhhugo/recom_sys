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
import gensim


spark = SparkSession.builder.appName("user cf on spark").master("local[8]").getOrCreate()

sc = spark.sparkContext

schema = StructType([StructField('userId', IntegerType(), True), 
            StructField('movieId', IntegerType(), True), 
            StructField('rating', LongType(), True), 
            StructField('timestamp', IntegerType(), True)])
ratings = spark.read.csv(r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\ml-20m\ratings.csv', header=True)

ratings = ratings.withColumn('rating', ratings['rating'].cast('int'))
ratings_rdd = ratings.select(['userId', 'movieId', 'rating']).rdd
ratings_rdd = ratings_rdd.sample(withReplacement=False, fraction=0.5, seed=2020)

print('user cf start......')
s = time.perf_counter()

createCombiner = lambda v: [v]
mergeValue = lambda agg, v: agg + [v]
mergeCombiners = lambda agg1, agg2: agg1 + agg2

user_items = ratings_rdd.map(lambda s: ('user_' + s['userId'], ('item_' + s['movieId'], s['rating']))).combineByKey(createCombiner, mergeValue, mergeCombiners).collectAsMap()
try:
    print('dumping')
    dump(user_items, r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\ml-20m\user_item_hist.pkl')
except KeyboardInterrupt:
    raise

print("cost time %s min" % ((time.perf_counter() - s) / 60))