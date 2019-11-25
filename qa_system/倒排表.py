import json
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from 问答系统 import read_corpus, cut_list, text_prepare


def top5_invidx(input_q):
    # 利用倒排表进行优化
    prepare_cut = cut_list(input_q)
    prepare_list = text_prepare(input_q)
    vector_martix = vector.transform(prepare_list).toarray()
    index_list = []
    for _ in prepare_cut:
        for input_word in _:
            if input_word in inverted_idx.keys():
                index_list.append(inverted_idx[input_word])

    index_list_flatten = list(set([i for j in index_list for i in j]))
    res = cosine_similarity(question_martix[index_list_flatten], vector_martix)
    top_5_index = np.argsort(res, axis=0)[-5:].squeeze().tolist()
    top_5_index.reverse()
    print('answer index', top_5_index)
    top_5_result = []
    for index in top_5_index:
        print('origin question:', question_list[index])
        print('origin answer:', answer_list[index])
        top_5_result.append(answer_list[index])
    return top_5_result


if __name__ == '__main__':
    file_path = './data/train-v2.0.json'
    question_list, answer_list = read_corpus(file_path)

    question_list = question_list[:10000]
    answer_list = answer_list[:10000]
    q_list = text_prepare(question_list)

    # 用TFIDF对句子进行表示
    vector = TfidfVectorizer()
    question_martix = vector.fit_transform(q_list).toarray()
    count_nonzero = np.count_nonzero(question_martix)
    a, b = question_martix.shape
    # 矩阵的稀疏度计算
    sparseness = count_nonzero / (a * b)
    print(sparseness)

    word_list_ = cut_list(question_list)
    word_list_question = [w for s in word_list_ for w in s]
    word_dict_question = dict(Counter(word_list_question))

    # 创建倒排表
    inverted_idx = {}
    for k, v in word_dict_question.items():
        if v < 1000:
            inverted_idx[k] = []
    # 在倒排表中添加索引
    # 这里会有重复的索引
    key_index = 0
    for str_ in word_list_:
        for word_ in str_:
            if word_ in inverted_idx.keys():
                inverted_idx[word_].append(key_index)
        key_index += 1

    input_question = 'How did Beyonce start becoming popular'
    result = top5_invidx([input_question])
    print(result)