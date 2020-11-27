import faust
import json
from keras_preprocessing.text import tokenizer_from_json
import konlpy
import math
import re
import requests
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
from faust.serializers import codecs

with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model.h5")
loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

okt = Okt()

stopwords = ['.',',','','의','가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를','으로', '자', '에', '와', '한', '하다']
hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
max_len = 200

label=["positive","normal","negative"]

with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


class MyCodec(codecs.Codec):

    def _dumps(self, obj):
        print(obj, type(obj))
        del obj['__faust']
        obj = json.dumps(obj, ensure_ascii=False)
        print(obj)
        return obj

    def _loads(self, s):
        return json.loads(s)
      
codecs.register('my_codec', MyCodec())

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
    return pad_new

def sentiment_predict(target):
    pad_new = preprocessing(target)
    score = loaded_model.predict(pad_new) # 예측
    return score

app = faust.App('nf-worker-1', broker='kafka://49.50.174.75:9092', broker_max_poll_record=3)

class Message(faust.Record):
    date: str
    collected_at: str
    id: str
    ip: str
    title: str
    body: str
    good: int
    bad: int
    is_reply: str
    positive_score: float = 0.0
    normal_score: float = 0.0
    negative_score: float = 0.0

nf_topic = app.topic('naver.finance.board.raw', value_type=Message)
target_topic = app.topic('naver.finance.board', value_type=Message)
target_url = "http://118.67.133.179:8888/target"


@app.agent(nf_topic)
async def finance_board(messages):
    async for msg in messages:
        sentence = msg.title+' '+msg.body
        print(sentence)
        score = sentiment_predict(sentence)
        msg.positive_score = round(float(score[0][2]),2)
        msg.normal_score = round(float(score[0][1]),2)
        msg.negative_score = round(float(score[0][0]),2)
        res = msg.asdict()
        print(res)
        requests.post(target_url, json=res)
        # yeild msg
       


if __name__ == '__main__':
    app.main()
