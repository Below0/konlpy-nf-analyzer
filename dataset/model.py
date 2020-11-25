#!/usr/bin/env python
# coding: utf-8

# In[2]:


import codecs
import pandas as pd

def saveCsv(df):
    df.to_csv(('./dataset.csv'),sep=',', na_rep='NaN',encoding='utf=8')

positive = []; negative = [];
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
del positive[-1]; del negative[-1]

posneg = positive+negative


# In[2]:


#-*- coding:utf-8 -*-

import os
import sys
import urllib.request
from urllib.parse import *
import requests
from bs4 import BeautifulSoup
#from kafka import KafkaProducer
import json
import re
import json
import datetime
from konlpy.tag import Okt

#with open('./config/config.json') as json_file:
#    config_data = json.load(json_file)

#kafka_ip = [config_data['kafka'] + ':9092']
#topic = 'naver.finance.board'

nlp = Okt()

dataset = []

j = 0

def remove_tag(content):
    cleanr = re.compile('<.*?>')
    result = re.sub(cleanr, '', content)
    return result


def get_bs_obj(url):
    result = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    bs_obj = BeautifulSoup(result.content, "html.parser")
    return bs_obj


class roomCrawler:
    main_url = 'https://finance.naver.com'

    def __init__(self, code='005930'):  # default code = 삼성전지
        self.room_url = self.main_url + '/item/board.nhn?code=' + code

    def detail_crawl(self, link):
        try:
            target_link = self.main_url + link
            bs = get_bs_obj(target_link)
            author_text = bs.find("span", class_="gray03").text
            ID, IP = author_text.split(" ")
            ID = ID[:-4]  # Erasing Masking

            og_date = bs.find("th", class_="gray03 p9 tah").text
            date_obj = datetime.datetime.strptime(og_date, '%Y.%m.%d %H:%M')
            date = date_obj.strftime('%Y-%m-%dT%H:%M:%S')

            title = bs.find("strong", class_="p15").text
            body = bs.find("div", id="body").text.replace("\r", " ")
            body = body.replace("\n", " ")
            total = title + ' ' + body
            post = total

        except Exception as err:
            post = None

        finally:
            return post

    def room_crawl(self, page=1):
        bs = get_bs_obj(self.room_url + '&page=' + str(page))
        lst = bs.find_all("td", class_="title")
        log_url = "http://127.0.0.1:8888/nfCrawlerResult"

        for i in range(len(lst) - 1, 0, -1):
            a = lst[i].find('a')
            link = a.get('href')
            post = self.detail_crawl(link)
            pos_cnt = 0
            neg_cnt = 0
            
            if post is None:
                continue
                
            for pn in range(len(posneg)):
                
                if pn < (len(positive)-1):
                    if post.find(posneg[pn]) != -1:
                        pos_cnt += 1
                if pn > (len(positive)-2):
                    if post.find(posneg[pn]) != -1:
                        neg_cnt += 1
                        
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
                "content":post
                           })

    def run(self):
        for i in range(1, 6):
            self.room_crawl(i)

if __name__ == "__main__":
    crawler = roomCrawler('005930')
    crawler.run()
    
df = pd.DataFrame(dataset)


# In[6]:


for item in dataset:
    print(item)


# In[62]:


import csv

dataset = []
contents = []
f = open('output.csv', 'r', encoding='utf-8')
rdf = csv.reader(f)
for line in rdf:
    contents.append(line[0]+' '+line[1])
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

        if pn < (len(positive)-1):
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
        "content":post
                   })
    
df = pd.DataFrame(dataset)
saveCsv(df)


# In[63]:


print(dataset)


# In[64]:


f = open('./dataset.csv', 'r', encoding='utf-8')
rdf = csv.reader(f)
for line in rdf:
    print(line)
f.close()


# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
dataset = pd.read_csv('./dataset.csv')


# In[66]:


dataset['label'].value_counts().plot(kind='bar')


# In[67]:


train_data = dataset[:1700]
test_data = dataset[1700:]
stopwords = ['.',',','','의','가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를','으로', '자', '에', '와', '한', '하다']


# In[72]:


import konlpy
import re
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer

hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')

X_train = []
for sentence in train_data['content']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('',word)
        if temp == '' or temp in stopwords:
            continue
        token_X.append(temp)
    X_train.append(token_X)
    
X_test = []
for sentence in test_data['content']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('',word)
        if temp == '' or temp in stopwords:
            continue
        token_X.append(temp)
    X_test.append(token_X)

print(X_test)


# In[73]:


max_words = 35000
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[75]:


print(len(X_train))


# In[76]:


import numpy as np

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if len(s) <= max_len:
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

y_train = []; y_test = []
for i in range(len(train_data['label'])):
    if train_data['label'].iloc[i] == 1:
        y_train.append([0,0,1])
    elif train_data['label'].iloc[i] == 0:
        y_train.append([0,1,0])
    elif train_data['label'].iloc[i] == -1:
        y_train.append([1,0,0])

for i in range(len(test_data['label'])):
    if test_data['label'].iloc[i] == 1:
        y_test.append([0,0,1])
    elif test_data['label'].iloc[i] == 0:
        y_test.append([0,1,0])
    elif test_data['label'].iloc[i] == -1:
        y_test.append([1,0,0])
        
y_train = np.array(y_train)
y_test = np.array(y_test)

print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

below_threshold_len(200, X_train)


# In[77]:


from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_len = 200 # 전체 데이터의 길이를 20로 맞춘다 
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=20,callbacks=[es,mc],batch_size=10, validation_split=0.1)


# In[78]:


print(model.evaluate(X_test, y_test)[1]*100)


# In[ ]:




