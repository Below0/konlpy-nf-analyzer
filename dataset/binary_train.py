import codecs
import csv
import pandas as pd
import numpy as np
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
from konlpy.tag import *
import konlpy
import re
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from keras.models import model_from_json

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if len(s) <= max_len:
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (cnt / len(nested_list)) * 100))

dataset = pd.read_csv('./dataset3.csv')

train_data = dataset[:11000]
test_data = dataset[11000:]
stopwords = ['.', ',', '', '의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

# In[72]:

okt = Mecab()

hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')

X_train = []
for sentence in train_data['content']:
    temp_X = okt.morphs(sentence)  # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('', word)
        if temp == '' or temp in stopwords:
            continue
        token_X.append(temp)
    X_train.append(token_X)

X_test = []
for sentence in test_data['content']:
    temp_X = okt.morphs(sentence)  # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('', word)
        if temp == '' or temp in stopwords:
            continue
        token_X.append(temp)
    X_test.append(token_X)

print('tokenizing complete!')
# In[73]:

max_words = 50000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

tokenizer_json = tokenizer.to_json()
with open('tokenizer3.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

y_train = []
y_test = []
for i in range(len(train_data['label'])):
    if train_data['label'].iloc[i] == 1:
        y_train.append([1])
    elif train_data['label'].iloc[i] == -1:
        y_train.append([0])

for i in range(len(test_data['label'])):
    if test_data['label'].iloc[i] == 1:
        y_test.append([1])
    elif test_data['label'].iloc[i] == -1:
        y_test.append([0])

y_train = np.array(y_train)
y_test = np.array(y_test)

print('리뷰의 최대 길이 :', max(len(l) for l in X_train))
print('리뷰의 평균 길이 :', sum(map(len, X_train)) / len(X_train))

below_threshold_len(200, X_train)

# In[77]:


from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_len = 200  # 전체 데이터의 길이를 20로 맞춘다
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_words, 128))

model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=1000, validation_split=0.1)

model_json = model.to_json()
with open("model3.json", "w") as json_file :
    json_file.write(model_json)

# In[78]:

with open("model3.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model.h5")
loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print(model.evaluate(X_test, y_test)[1] * 100)