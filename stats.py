import numpy as np
import json
import re
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
import pickle
import string


sc = SparkContext()
sqlContext = SQLContext(sc)
data = sc.textFile("/datasets/tweets-leon")

"""Select tweet by filtering"""
def selection_tweet(tweet):
    array = tweet.split("\t")
    if (len(array)==5):
    	if (array[0] == 'en' and array[2][-4:]=='2014'):
        	return True
    return False

data_2014 = data.filter(selection_tweet)

tweet_count = data_2014.map(lambda tweet: (tweet.split('\t')[2].split(' ')[1], 1)) \
				  	   .reduceByKey(lambda a, b: a + b) \
				  	   .sortBy(lambda wc: -wc[1])

counts = sqlContext.createDataFrame(tweet_count.map(lambda wc: Row(month=wc[0], count=wc[1])))

counts.show()

# save to json
counts.write.json("tweets_count.txt")