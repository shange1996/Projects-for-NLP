from nltk.corpus import reuters
import numpy as np


def generate_candidates(word):
    """
    :param word:给定的输入，错误的输入
    :return: 返回所有候选集合， 编辑距离为1的单词
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    split = [(word[:i], word[i:]) for i in range(len(word)+1)]

    # insert操作
    insert = [l+r+c for l, r in split for c in letters]
    # delete操作
    delete = [l+r[1:] for l, r in split if r]
    # replace
    replace = [l+c+r[1:] for l, r in split for c in letters if r]

    candidate = set(insert+delete+replace)
    return [w for w in candidate if w in vocab]


def generate_candidates_two(word):
    """
        :param word:给定的输入，错误的输入
        :return: 返回所有候选集合， 编辑距离为2的单词
        """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    # 首先生成编辑距离为1的单词
    split = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # insert操作
    insert = [l + r + c for l, r in split for c in letters]
    # delete操作
    delete = [l + r[1:] for l, r in split if r]
    # replace
    replace = [l + c + r[1:] for l, r in split for c in letters if r]
    candi = list(set(insert + delete + replace))

    # 从编辑距离为1的单词中再进行一次编辑
    candi_two = []
    for c in candi:
        temp = generate_candidates(c)
        candi_two.append(temp)
    candi_two = [w for _ in candi_two for w in _ if w in vocab]
    return list(set(candi_two))


# 利用bigram模型得到language model
def generate_bigram():
    term_count = {}
    bigram_count = {}
    for doc in corpus:
        doc = ['<s>'] + doc
        for i in range(1, len(doc)-1):
            bigram = doc[i:i+2]
            bigram = ' '.join(bigram)
            if doc[i] in term_count:
                term_count[doc[i]] += 1
            else:
                term_count[doc[i]] = 1

            if bigram in bigram_count:
                bigram_count[bigram] += 1
            else:
                bigram_count[bigram] = 1
    return term_count, bigram_count


def spelling_check(word, channel_model, term_count, bigram_count, s):
    """

    :param word: 错误的单词
    :param channel_model: 概率
    :param term_count: unigram_count
    :param bigram_count:
    :return:
    """
    # 生成候选集合
    candidates = generate_candidates(word)
    v = len(term_count.keys())
    if len(candidates) < 3:
        # 生成编辑距离为2的单词
        candidates = generate_candidates_two(word)
    if not candidates:
        # 如果仍然没有候选词，可以认为输入没有错误
        return
    probs = []
    for candi in candidates:
        # 得到channel model的概率
        prob = 0
        if candi in channel_model and word in channel_model[candi]:
            prob += np.log(channel_model[candi][word])
        else:
            prob += np.log(0.00001)

        # 得到bigram的概率
        idx = s.index(word)
        word_pre, word_next = '', ''
        if idx > 1:
            word_pre = s[idx - 1]
        if idx < len(s) - 1:
            word_next = s[idx + 1]

        bi_pre = ' '.join([word_pre, candi])
        bi_next = ' '.join([candi, word_next])
        if bi_pre in bigram_count:
            prob_pre = (np.log(bigram_count[bi_pre]) + 1.0)/(term_count[word_pre]+v)
        else:
            prob_pre = np.log(1.0/v)
        if bi_next in bigram_count:
            prob_next = (np.log(bigram_count[bi_next]) + 1.0)/(term_count[word_next]+v)
        else:
            prob_next = np.log(1.0/v)
        prob += 0.5*(prob_pre+prob_next)
        probs.append(prob)
    max_idx = probs.index(max(probs))
    print(word, candidates, candidates[max_idx])


if __name__ == "__main__":
    with open('./data/vocab.txt', 'r', encoding='utf-8') as f:
        vocab = set([line.strip() for line in f.readlines()])

    # 所有分类
    categories = reuters.categories()
    # 指定分类的句子
    corpus = reuters.sents(categories=categories)
    term_count, bigram_count = generate_bigram()
    # 得到了channel model
    channel_prob = {}
    with open('./data/spell-errors.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split(':')
            correct = items[0]
            mistakes = [c.strip('.') for c in items[1].strip().split(',')]
            channel_prob[correct] = {}
            for mis in mistakes:
                channel_prob[correct][mis] = 1.0/len(mistakes)

    with open('./data/testdata.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split('\t')[2]
            s = items.split()
            s = [c.strip('.,') for c in s]
            for word in s:
                if word not in vocab:
                    spelling_check(word, channel_prob, term_count, bigram_count, s)