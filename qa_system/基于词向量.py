import json
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

path = './data/glove.6B.100d.txt'
with open(path, 'r', encoding='utf-8') as f:
    data = f.readlines()

vocab = []
emb = []
for line in data:
    row = line.strip().split(' ')
    vocab.append(row[0])
    emb.append(row[1:])

emb = np.array(emb)
print(emb.shape)


def get_words_vec(words):
    """
    :param words: str
    """
    glove_index = []
    for word in words.split(' '):
        if word in vocab:
            index = vocab.index(word)
            glove_index.append(index)
    return emb[glove_index].astype('float').sum()/len(words.split(' '))


def read_corpus(p):
    q_list = []
    a_list = []
    with open(p, 'r', encoding='utf-8') as file:
        temp_data = json.load(file)['data']
        for para in temp_data:
            paragraphs = para['paragraphs']
            for _ in paragraphs:
                qas = _['qas']
                for qa_dict in qas:
                    if 'plausible_answers' in qa_dict:
                        q_list.append(qa_dict['question'])
                        a_list.append(qa_dict['plausible_answers'][0]['text'])
                    else:
                        q_list.append(qa_dict['question'])
                        a_list.append(qa_dict['answers'])
    assert len(q_list) == len(a_list)
    return q_list, a_list


def cut_list(inputs):
    temp = []
    for _ in inputs:
        temp.append(nltk.word_tokenize(_))
    return temp


def text_prepare(inputs_list):
    word_list = cut_list(inputs_list)
    flatten_list = [w for s in word_list for w in s]
    word_dict = dict(Counter(flatten_list))
    low_frequency = []
    le = WordNetLemmatizer()
    for k, v in word_dict.items():
        if v < 1:
            low_frequency.append(k)
    new_list = []
    for s in word_list:
        temp_str = ''
        for word in s:
            # 去除标点符号和低频
            if word in string.punctuation or word in low_frequency:
                continue
            # stemming(包括了小写)
            word = le.lemmatize(word.lower())
            # 转换为数字
            if word.isdigit():
                word = word.replace(word, '#number')
            if word not in stopwords.words('english'):
                temp_str += word + ' '
        new_list.append(temp_str)

    return new_list


# 计算所有问题的向量
def all_question_matrix(input_list):
    """
    :param input_list: [str]
    """
    q_matrix = []
    for temp_str in input_list:
        q_matrix.append(get_words_vec(temp_str))

    return np.array(q_matrix)


def create_inverted(input_list):
    """
    :param input_list: [str]
    """
    word_list_ = cut_list(input_list)
    word_list_question = [w for s in word_list_ for w in s]
    word_dict_question = dict(Counter(word_list_question))
    # 创建倒排表
    inverted_idx = {}
    for k, v in word_dict_question.items():
        if v < 1000:
            inverted_idx[k] = []
    key_index = 0
    for str_ in word_list_:
        for word_ in str_:
            if word_ in inverted_idx.keys():
                inverted_idx[word_].append(key_index)
        key_index += 1
    return inverted_idx


# 利用倒排表进行优化
def top5result(input_q, inverted_idx, q_matrix):
    input_q = text_prepare(input_q)
    prepare = cut_list(input_q)
    print(prepare)
    # 先找到可能存在的回答
    q_index_list = []
    for str_ in prepare:
        for word_ in str_:
            if word_ in inverted_idx.keys():
                q_index_list.append(inverted_idx[word_])
    index_list_flatten = list(set([i for j in q_index_list for i in j]))
    q_matrix_inverted = q_matrix[index_list_flatten]
    # 得到当前输入的向量
    for str_ in input_q:
        input_matrix = get_words_vec(str_)
    value = np.abs(q_matrix_inverted - input_matrix)
    top5index = np.argsort(value, axis=0)[-5:].squeeze().tolist()
    print(top5index)
    top5index.reverse()
    top_5_result = []
    for index in top5index:
        print('origin question:', question_list[index])
        print('origin answer:', answer_list[index])
        top_5_result.append(answer_list[index])
    return top_5_result


if __name__ == '__main__':
    file_path = './data/train-v2.0.json'
    question_list, answer_list = read_corpus(file_path)

    question_list = question_list[:10000]
    answer_list = answer_list[:10000]
    question_list = text_prepare(question_list)
    question_matrix = all_question_matrix(question_list)
    inverted_dict = create_inverted(question_list)

    input_question = 'Beyonce start popular'
    reslut = top5result([input_question], inverted_dict, question_matrix)
