import codecs
import pandas as pd
import csv
from konlpy.tag import *
from collections import Counter

mecab = Mecab()

def saveCsv(df, name):
    df.to_csv(('./'+name+'.csv'), sep=',', na_rep='NaN', encoding='utf=8')

positive = [];
negative = [];
posneg = []

pos = codecs.open("./positive_words_self.txt", 'rb', encoding='UTF-8')

while True:
    line = pos.readline()
    line = line.replace('\n', '')
    positive.append(line)

    if not line:
        break
pos.close()

neg = codecs.open("./negative_words_self.txt", 'rb', encoding='UTF-8')
while True:
    line = neg.readline()
    line = line.replace('\n', '')
    negative.append(line)
    if not line:
        break

neg.close()
del positive[-1];
del negative[-1]

posneg = positive + negative

dataset = []
contents = []
f = open('output.csv', 'r', encoding='utf-8')
rdf = csv.reader(f)
for line in rdf:
    contents.append(line[2] + ' ' + line[3])
f.close()

del contents[0]

# 전체 필터 추가
filters = []
pos = codecs.open("filter.txt", 'rb', encoding='UTF-8')
while True:
    line = pos.readline()
    line = line.replace('\n', '').replace('\r', '')
    filters.append(line)
    if not line:
        break
pos.close()
del filters[-1]

#정치글 삭제
j = 0
for i in range(len(contents)):
    idx = i + j
    for word in filters:
        if contents[idx].find(word) > -1:
            del contents[idx]
            j -= 1
            break

pos_list = []
neg_list = []
normal_list = []

for post in contents:
    pos_cnt = 0
    neg_cnt = 0

    for pn in range(len(posneg)):
        index = -1
        score = 0
        while True:
            index = post.find(posneg[pn], index + 1)
            if index == -1:
                break
            else:
                score += 1

        if pn < (len(positive) - 1):
            pos_cnt += score
        else:
            neg_cnt += score

    total = pos_cnt + neg_cnt

    try:
        ratio = pos_cnt / total
    except Exception as err:
        ratio = 0

    tokens = mecab.morphs(post)

    if ratio >= 0.8:
        for token in tokens:
            pos_list.append(token)
    elif ratio <= 0.2:
        for token in tokens:
            neg_list.append(token)
    else:
        for token in tokens:
            normal_list.append(token)


neg_count = Counter(neg_list)
pos_count = Counter(pos_list)
normal_count = Counter(normal_list)

neg_words = neg_count.most_common(100)
pos_words = pos_count.most_common(100)

common_words = []
only_words = []
pw = []

for w in pos_words:
    pw.append(w[0])

for k,v in neg_words:
    if k in pw:
        common_words.append(k)

df = pd.DataFrame(common_words)
saveCsv(df,'common')


