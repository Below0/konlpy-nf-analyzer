
import json
from keras_preprocessing.text import tokenizer_from_json
import konlpy
import re
import csv
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


weight_name = "rmsprop.h5"
model_json_name = "model2.json"
tokenizer_name = "tokenizer2.json"

with open(model_json_name, "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(weight_name)
#loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
okt = Mecab()
stop_words = []
with open('common.csv', 'r', encoding='utf-8') as f:
    rdf = csv.reader(f)
    for line in rdf:
        stop_words.append(line[1])
    f.close()

hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
max_len = 200

label=["부정","중립","긍정"]

with open('tokenizer2.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def preprocessing(sentence):
    temp_X = okt.morphs(sentence) # 토큰화
    token_X = []
    for word in temp_X:
        temp = hangul.sub('',word)
        if temp == '' or temp in stop_words:
            continue
        token_X.append(temp)
    print(token_X)
    encoded = tokenizer.texts_to_sequences([token_X])
    pad_new = pad_sequences(encoded, maxlen = max_len)
    return pad_new

def sentiment_predict(target):
    score = loaded_model.predict(preprocessing(target)) # 예측
    print(score)
    print(label[np.argmax(score)])

# In[10]:

sentiment_predict("삼성전자덕분에 부자되고 너무 행복합니다 화이팅")


# In[ ]:




