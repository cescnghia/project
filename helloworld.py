import numpy as np
import json
import re
from pyspark.sql import *
from pyspark import SparkContext, SQLContext
from pyspark.ml.feature import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import functions as F
import pickle

sc = SparkContext()
sqlContext = SQLContext(sc)
data = sc.textFile("/datasets/tweets-leon")


def selection_tweet(tweet): return len(tweet.split("\t")) == 5

def encode_tweet(tweet): return [t.encode("utf8") for t in tweet.split("\t")]

"""Data in english"""
data = data.filter(selection_tweet)
en_data = data.filter(lambda x : x[:2]=='en')

"""Encode UTF-8"""
en_data = en_data.map(encode_tweet)

"""Take only ID and CONTENT of a tweet"""
tweets = en_data.map(lambda tweet : Row(id=tweet[1], sentence=tweet[4]))

"""Create DF"""
df_tweets = sqlContext.createDataFrame(tweets)

"""Tokenization"""
regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="raw", pattern="\\W")
regexTokenized = regexTokenizer.transform(df_tweets)

print('Finish Tokenization phase')

"""Remove Stop-words"""
remover = StopWordsRemover(inputCol="raw", outputCol="filtered")
removed_stopwords = remover.transform(regexTokenized)

print('Finish Remove Stop-words phase')

"""Lemmatization"""
#TODO

"""IF-IDF"""
cv = CountVectorizer(inputCol="filtered", outputCol="vectors")
count_vectorizer_model = cv.fit(removed_stopwords)
tf = count_vectorizer_model.transform(removed_stopwords)

idf = IDF(inputCol="vectors", outputCol="tfidf")
idfModel = idf.fit(tf)
tfidf = idfModel.transform(tf)

print('Finish IF-IDF phase')

"""Features extraction with LDA"""
nbTopics=70
n_terms=15

corpus = tfidf.select(F.col('id').cast("long"), 'tfidf').rdd.map(lambda x: [x[0], x[1]])
ldaModel = LDA.train(corpus, k=nbTopics)

topics = ldaModel.describeTopics(maxTermsPerTopic=n_terms)
vocabulary = count_vectorizer_model.vocabulary

"""Store result"""
with open("topics.pickle", "wb") as f:
    pickle.dump(topics, f)
with open("vocabulary.pickle", "wb") as f:
    pickle.dump(vocabulary, f)

file = open("results.txt","w")
for topic in range(len(topics)):
    print("topic {} : ".format(topic))
    file.write("topic {} : \n".format(topic)) 
    words = topics[topic][0]
    scores = topics[topic][1]
    for word in range(len(words)):
        file.write("{} - {}\n".format(vocabulary[words[word]], scores[word]))
        print(vocabulary[words[word]], "-", scores[word])
file.close()

#print ('The number of tweets tolerant: ',count_tolerant)