import numpy as np

word2idx, idx2word = {}, {}
tag2idx, idx2tag = {}, {}

with open('./input/traindata.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

    for line in data:
        items = line.strip().split('/')
        word = items[0].strip()
        tag = items[1].strip()
        if word not in word2idx:
            word2idx[word] = len(word2idx)
            idx2word[len(idx2word)] = word
        if tag not in tag2idx:
            tag2idx[tag] = len(tag2idx)
            idx2tag[len(idx2tag)] = tag

print(tag2idx)
print(idx2tag)
n_tag = len(tag2idx)
n_word = len(word2idx)

con = np.zeros(n_tag)
like_hood = np.zeros(shape=(n_tag, n_word))
bi_gram = np.zeros(shape=(n_tag, n_tag))

prev_tag = ''
for line in data:
    items = line.strip().split('/')
    word = items[0].strip()
    tag = items[1].strip()
    if prev_tag == '':
        con[tag2idx[tag]] += 1
        like_hood[tag2idx[tag]][word2idx[word]] += 1
    else:
        like_hood[tag2idx[tag]][word2idx[word]] += 1
        bi_gram[tag2idx[prev_tag]][tag2idx[tag]] += 1
    if word == '.':
        prev_tag = ''
    else:
        prev_tag = tag

con = (con+1)/(np.sum(con)+n_tag)
for i in range(n_tag):
    like_hood[i] = (like_hood[i]+1)/(like_hood[i].sum()+n_word)
    bi_gram[i] = (bi_gram[i] + 1)/(bi_gram[i].sum()+n_tag)


# 利用动态规划
def viterbi(sentence, con, like_hood, bi_gram):
    x = [word2idx[w] for w in sentence.strip().split(' ')]
    n = len(x)
    dp = np.zeros(shape=(n, n_tag))
    pos = np.zeros(shape=(n, n_tag), dtype=np.int)

    for i in range(n_tag):
        dp[0][i] = np.log(con[i]) + np.log(like_hood[i][x[0]])
    for i in range(1, n):
        for j in range(n_tag):
            dp[i][j] = -9999
            for k in range(n_tag):
                score = dp[i-1][k] + np.log(like_hood[j][x[i]]) + np.log(bi_gram[k][j])
                if score > dp[i][j]:
                    dp[i][j] = score
                    pos[i][j] = k
    # 找出最好的词性组合
    best_seq = [0] * n
    # 找到最后一个的位置
    best_seq[n-1] = np.argmax(dp[n-1])
    # 找到其他的位置
    for i in range(n-2, -1, -1):
        best_seq[i] = pos[i+1][best_seq[i+1]]
    print(best_seq)
    best_tag = [idx2tag[_] for _ in best_seq]
    print(best_tag)


viterbi('The new ad plan from Newsweek', con, like_hood, bi_gram)
print('The new ad plan from Newsweek')