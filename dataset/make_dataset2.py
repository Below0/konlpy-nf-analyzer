import codecs
import pandas as pd
import csv
from konlpy.tag import *

mecab = Mecab()

def saveCsv(df):
    df.to_csv(('./dataset.csv'), sep=',', na_rep='NaN', encoding='utf=8')


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

    if ratio >= 0.8:
        pos_value = 1
    elif ratio <= 0.2:
        pos_value = -1
    else:
        pos_value = 0

    dataset.append({
        "label": pos_value,
        "content": post
    })

df = pd.DataFrame(dataset)
saveCsv(df)
'''
for post in contents:
    token = mecab.morphs(post)
    print(token)
    pos_cnt = 0
    neg_cnt = 0

    for t in token:
        idx = posneg.index(t)

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

    if ratio >= 0.6:
        pos_value = 1
    elif ratio <= 0.4:
        pos_value = -1
    else:
        pos_value = 0

    dataset.append({
        "label": pos_value,
        "content": post
    })

df = pd.DataFrame(dataset)
saveCsv(df)
'''
