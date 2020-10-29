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
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.appName("tag base on spark").master("local[8]").getOrCreate()

sc = spark.sparkContext

schema = StructType([StructField('userId', IntegerType(), True), 
            StructField('movieId', IntegerType(), True), 
            StructField('rating', LongType(), True), 
            StructField('timestamp', IntegerType(), True)])
tags = spark.read.csv(r'D:\Users\hao.guo\deepctr\recsys\movielen\ml-20m\tags.csv', header=True)

index = StringIndexer(inputCol='tag', outputCol='tagid')
model = index.fit(tags)
tags = model.transform(tags)
tags = tags.withColumn('tagid', tags['tagid'].cast('int'))
tags_rdd = tags.select(['userId', 'movieId', 'tagid']).rdd

train_rdd, test_rdd = tags_rdd.randomSplit([0.7, 0.3], seed=2020)
train_rdd = train_rdd.cache()
test_rdd = test_rdd.cache()

train_rdd = train_rdd.map(lambda s: (s, 1)).
