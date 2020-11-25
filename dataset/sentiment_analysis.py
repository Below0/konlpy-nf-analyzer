#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json
from keras_preprocessing.text import tokenizer_from_json
import konlpy
import re
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
from keras.models import model_from_json
import numpy as np
from konlpy.tag import Okt
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model.h5")
loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

okt = Okt()

stopwords = ['.',',','','의','가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를','으로', '자', '에', '와', '한', '하다']
hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
max_len = 200

label=["긍정","중립","부정"]

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def preprocessing(sentence):
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('',word)
        if temp == '' or temp in stopwords:
            continue
        token_X.append(temp)
    encoded = tokenizer.texts_to_sequences([token_X])
    pad_new = pad_sequences(encoded, maxlen = max_len)

def sentiment_predict(target):
    target = okt.morphs(target, stem=True) # 토큰화
    target= [word for word in target if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([target]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    score = loaded_model.predict(pad_new) # 예측
    print(label[np.argmax(score)])


# In[10]:


sentiment_predict("삼성전자 화이팅")


# In[ ]:




