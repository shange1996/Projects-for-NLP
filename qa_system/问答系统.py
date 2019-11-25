import json
import nltk
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def read_corpus(path):
    q_list = []
    a_list = []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
        for para in data:
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


def plot_word():
    word_list_question = cut_list(question_list)
    word_list_question = [w for s in word_list_question for w in s]
    word_dict_question = dict(Counter(word_list_question))
    word_total_question = len(word_dict_question)
    print('问题中总共出现了{}个单词'.format(word_total_question))
    y = []
    for i in word_dict_question:
        y.append(word_dict_question[i])

    plt.figure()
    plt.plot(sorted(y, reverse=True))
    plt.show()


#  文本预处理
#   1. 停用词过滤 （去网上搜一下 "english stop words list"，会出现很多包含停用词库的网页，或者直接使用NLTK自带的）
#   2. 转换成lower_case： 这是一个基本的操作
#   3. 去掉一些无用的符号： 比如连续的感叹号！！！， 或者一些奇怪的单词。
#   4. 去掉出现频率很低的词：比如出现次数少于10,20....
#   5. 对于数字的处理： 分词完只有有些单词可能就是数字比如44，415，把所有这些数字都看成是一个单词，这个新的单词我们可以定义为 "#number"
#   6. stemming（利用porter stemming): 因为是英文，所以stemming也是可以做的工作
def text_prepare(inputs_list):
    word_list = cut_list(inputs_list)
    flatten_list = [w for s in word_list for w in s]
    word_dict = dict(Counter(flatten_list))
    low_frequency = []
    p = nltk.stem.porter.PorterStemmer()
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
            word = p.stem(word)
            # 转换为数字
            if word.isdigit():
                word = word.replace(word, '#number')
            if word not in stopwords.words('english'):
                temp_str += word + ' '
        new_list.append(temp_str)

    return new_list


def top5result(input_q, q_list, a_list):
    '''
    先对输入进行文本预处理，再进行tfidf，然后计算5个最相近的问题，返回答案
    :param input_q:  List[str]
    :return:
    '''
    prepare_list = text_prepare(input_q)
    vector_martix = vector.transform(prepare_list).toarray()
    res = cosine_similarity(question_martix, vector_martix)
    top_5_index = np.argsort(res, axis=0)[-5:].squeeze().tolist()
    top_5_index.reverse()
    print('answer index', top_5_index)
    top_5_result = []
    for index in top_5_index:
        print('origin question:', q_list[index])
        print('origin answer:', a_list[index])
        top_5_result.append(a_list[index])
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

    input_question = 'How did Beyonce start becoming popular'
    print('input_question: ', input_question)
    reply_answer = top5result([input_question], question_list, answer_list)
    # print(reply_answer)