import numpy as np
import json
import re
import matplotlib.pyplot as plt
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
truncate_df = removed_stopwords.drop('sentence')

"""TF"""
"Parameter minDF: this term have to be appear in a specific nb of docs (fraction of nb of docs)"
cv = CountVectorizer(inputCol="filtered", outputCol="vectors", minDF=.2)
count_vectorizer_model = cv.fit(truncate_df)
tf = count_vectorizer_model.transform(truncate_df)

voca = count_vectorizer_model.vocabulary
vocabulary = sc.parallelize(voca)
vocabulary_df = sqlContext.createDataFrame(vocabulary.map(Row))
vocabulary_df.write.json("vocabulary.txt")

truncate_df = tf.drop('filtered')
truncate_df.save('tf', mode='overwrite')