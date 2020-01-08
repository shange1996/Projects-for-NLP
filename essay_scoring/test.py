from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import collections

word_dict = collections.defaultdict(lambda: 0)
t = ['apple fruit apple', 'love apple', 'apple be']
vec = CountVectorizer(ngram_range=(1, 3))
print(vec.fit_transform(t).toarray())
print(vec.get_feature_names())