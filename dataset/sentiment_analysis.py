#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json
from keras_preprocessing.text import tokenizer_from_json
import konlpy
import re
from konlpy.tag import Mecab
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from keras.models import model_from_json
import numpy as np
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


with open("model2.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model.h5")
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

okt = Mecab()

stopwords = ['.',',','','의','가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를','으로', '자', '에', '와', '한', '하다']
hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
max_len = 200

label=["긍정","중립","부정"]

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def preprocessing(sentence):
    temp_X = okt.morphs(sentence) # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('',word)
        if temp == '' or temp in stopwords:
            continue
        token_X.append(temp)
    print(token_X)
    encoded = tokenizer.texts_to_sequences([token_X])
    pad_new = pad_sequences(encoded, maxlen = max_len)
    return pad_new

def sentiment_predict(target):
    score = loaded_model.predict(preprocessing(target)) # 예측
    print(score)


# In[10]:

sentiment_predict("삼성전자 망한다.")


# In[ ]:




