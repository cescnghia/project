import numpy as np
import json
import re
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
from pyspark.ml.feature import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import functions as F
import pickle
import string


sc = SparkContext()
sqlContext = SQLContext(sc)

removed_stopwords = sqlContext.read.load('removed_stopwords')

filtered_column = removed_stopwords.map(lambda x : x['filtered'])

count_rrd = filtered_column.flatMap(lambda x : x) \
                           .map(lambda x : (x, 1)) \
                           .reduceByKey(lambda a, b: a + b) \
                           .sortBy(lambda wc: -wc[1])

term_counts = sqlContext.createDataFrame(count_rrd.map(lambda wc: Row(term=wc[0], count=wc[1])))

# save to json
term_counts.write.json("term_counts.txt")
