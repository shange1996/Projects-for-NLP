import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


dataframe = pd.read_csv('essays_and_scores.csv', encoding = 'latin-1')

data = dataframe[['essay_set','essay','domain1_score']].copy()


def get_count_vectors(essays):
    vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')

    count_vectors = vectorizer.fit_transform(essays)

    feature_names = vectorizer.get_feature_names()

    return feature_names, count_vectors

feature_names_cv, count_vectors = get_count_vectors(data[data['essay_set'] == 1]['essay'])

X_cv = count_vectors.toarray()

y_cv = data[data['essay_set'] == 1]['domain1_score'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X_cv, y_cv, test_size = 0.3)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)

# The coefficients
print('LinearRegression Coefficients: \n', linear_regressor.coef_)
# The mean squared error
print("LinearRegression Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('LinearRegression MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print('Ridge Coefficients: \n', ridge.coef_)
print("Ridge Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Ridge MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)


def sentence_to_wordlist(raw_sentence):
    clean_sentence = re.sub("[^a-zA-Z0-9]", " ", raw_sentence)
    tokens = nltk.word_tokenize(clean_sentence)
    return tokens


def tokenize(essay):
    stripped_essay = essay.strip()

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)

    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))
    return tokenized_sentences

# 每个单词平均有几个字母
def avg_word_len(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    return sum(len(word) for word in words) / len(words)

# 文章单词总数
def word_count(essay):
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)
    return len(words)


# 文章字母总数
def char_count(essay):
    clean_essay = re.sub(r'\s', '', str(essay).lower())
    return len(clean_essay)

# 文章句子总数
def sent_count(essay):
    sentences = nltk.sent_tokenize(essay)
    return len(sentences)

# 统计词形还原后的单词总数
def count_lemmas(essay):
    tokenized_sentences = tokenize(essay)
    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    lemma_count = len(set(lemmas))

    return lemma_count

# 文章中错误单词总数
def count_spell_error(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    # big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg
    #         and lists of most frequent words from Wiktionary and the British National Corpus.
    #         It contains about a million words.
    data = open('big.txt').read()

    words_ = re.findall('[a-z]+', data.lower())

    word_dict = collections.defaultdict(lambda: 0)

    for word in words_:
        word_dict[word] += 1

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()

    for word in words:
        if not word in word_dict:
            mispell_count += 1

    return mispell_count

# 统计名词，动词，形容词和副词的数量
def count_pos(essay):
    tokenized_sentences = tokenize(essay)

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1

    return noun_count, adj_count, verb_count, adv_count

# 统计文章中每篇文章的词频
def get_count_vectors(essays):
    vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')

    count_vectors = vectorizer.fit_transform(essays)

    feature_names = vectorizer.get_feature_names()

    return feature_names, count_vectors


def extract_features(data):
    features = data.copy()

    features['char_count'] = features['essay'].apply(char_count)
    features['word_count'] = features['essay'].apply(word_count)
    features['sent_count'] = features['essay'].apply(sent_count)
    features['avg_word_len'] = features['essay'].apply(avg_word_len)
    features['lemma_count'] = features['essay'].apply(count_lemmas)
    features['spell_err_count'] = features['essay'].apply(count_spell_error)
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(
        *features['essay'].map(count_pos))

    return features


features_set1 = extract_features(data[data['essay_set'] == 1])
print(features_set1)


features_set1.plot.scatter(x = 'char_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'word_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'sent_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'avg_word_len', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'lemma_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'spell_err_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'noun_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'adj_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'verb_count', y = 'domain1_score', s=10)
features_set1.plot.scatter(x = 'adv_count', y = 'domain1_score', s=10)

X = np.concatenate((features_set1.iloc[:, 3:].as_matrix(), X_cv), axis = 1)
y = features_set1['domain1_score'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)

# The coefficients
print('LinearRegression Coefficients: \n', linear_regressor.coef_)
# The mean squared error
print("LinearRegression Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('LinearRegression MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
print('Ridge Coefficients: \n', ridge.coef_)
print("Ridge Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Ridge MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)


ridge_ = Ridge()
param_alpha = {'alpha': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]}
grid = GridSearchCV(estimator=ridge_, param_grid=param_alpha)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
print("Grid Ridge Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Grid Ridge MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)


params = {'n_estimators':[100, 1000], 'max_depth':[2], 'min_samples_split': [2], 'learning_rate':[3, 1, 0.1, 0.3], 'loss': ['ls']}
gbdt = ensemble.GradientBoostingRegressor()
grid = GridSearchCV(estimator=gbdt,param_grid=params)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
print("Grid GBDT Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Grid GBDT Variance score: %.2f' % ridge.score(X_test, y_test))
print('Grid GBDT MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)

rf = ensemble.RandomForestRegressor()
params = {'n_estimators': [10, 50, 100], 'max_depth':[10, 50, 100], 'max_features':[2, 5, 10]}
grid = GridSearchCV(estimator=rf, param_grid=params)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)
print("Grid RF Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Grid RF Variance score: %.2f' % ridge.score(X_test, y_test))
print('Grid RF MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)


model = xgb.XGBRegressor()
params = {'n_estimators': [100, 150, 160], 'learning_rate': [0.1, 0.5, 1.0]}
grid = GridSearchCV(estimator=model, param_grid=params)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print("Grid xgboost Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Grid xgboost MAPE:%.2f' % np.average(np.abs((y_test-y_pred)/y_test)))
print('-' * 50)