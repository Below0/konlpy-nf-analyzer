import codecs
import pandas as pd
import csv


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