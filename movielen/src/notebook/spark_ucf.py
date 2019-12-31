# -*- coding: utf-8 -*-
import findspark
import gc
from joblib import dump, load   
findspark.init()
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
spark = SparkSession.builder.appName("user cf on spark").master("local[8]").getOrCreate()

sc = spark.sparkContext

item_users = load(r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\data\item_users.pkl')
users = load(r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\data\users.pkl')

item_users_rdd = sc.parallelize(list(item_users.items())).persist()
users_rdd = sc.parallelize(users).persist()
del item_users, users
gc.collect()

from pyspark.sql.types import StructField, StringType, IntegerType, FloatType, StructType
schema = StructType([StructField('userId', IntegerType(), True), 
                     StructField('movieId', IntegerType(), True), 
                     StructField('rating', FloatType(), True), 
                     StructField('timestamp', IntegerType(), True)])
ratings = spark.read.csv(r'D:\Users\hao.guo\比赛代码提炼\推荐系统\movielen\ml-20m\ratings.csv', schema=schema, header=True)

ratings = ratings.withColumn('userId_new', 'user_' + ratings['userId'].cast(StringType()))
ratings = ratings.withColumn('movieId_new', 'item_' + ratings['movieId'].astype(StringType()))
from pyspark.sql.functions import utf
slen = udf(lambda s: len(s), IntegerType())
df.select(slen("name").alias("slen_name"))

def UserSimilarity(item_users_rdd, users_rdd):
    users_dict = {u: 0 for u in users}
    C = users_rdd.map(lambda s: {s: })
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