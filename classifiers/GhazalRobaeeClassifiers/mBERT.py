import tensorflow as tf
import pandas as pd
import hazm
import re
import csv
import glob
import os
import numpy as np
from hazm import stopwords_list
from transformers import TFAutoModel, TFAlbertForSequenceClassification
from transformers import AutoConfig, AutoTokenizer
from transformers import BertConfig, BertTokenizer
from transformers import TFAutoModel, TFBertForSequenceClassification



def cleaning(text):
    text = text.strip()
    # normalizing
    text = normalizer.normalize(text)

    # replacing all spaces,hyphens,  with white space
    space_pattern = r"[\xad\ufeff\u200e\u200d\u200b\x7f\u202a\u2003\xa0\u206e\u200c\x9d\]]"
    space_pattern = re.compile(space_pattern)
    text = space_pattern.sub(" ", text)

    # let's delete the un-required elements
    deleted_pattern = r"(\d|[\|\[]]|\"|'ٍ|[0-9]|¬|[a-zA-Z]|[؛“،,”‘۔’’‘–]|[|\.÷+\:\-\?»\=\{}\*«_…\؟!/ـ]|[۲۹۱۷۸۵۶۴۴۳]|[\\u\\x]|[\(\)]|[۰'ٓ۫'ٔ]|[ٓٔ]|[ًٌٍْﹼ،َُِّ«ٰ»ٖء]|\[]|\[\])"
    deleted_pattern = re.compile(deleted_pattern)
    text = deleted_pattern.sub("", text).strip()


    # removing wierd patterns
    wierd_pattern = re.compile("["
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        # u"\0x06F0-\0x06F9"
        u"\u200d"
        u"\u200c"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)

    text = wierd_pattern.sub(r'', text)
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    return text

def stop_word_importer(file_name):# importing persian stopwords
    with open(file_name, 'r', encoding="utf8") as myfile:
        stop_words = myfile.read().replace('\n', ' ').replace("\u200c","").replace("\ufeff","").replace("."," ").split(' ')# a list of stop words
    return stop_words

def removeStopWords(text):
    stop_words = stop_word_importer('C:\\Users\\Golnoosh\\Desktop\\stop_words.txt') 
    text = ' '.join([word for word in text.split() if word not in stopwords_list()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def prep_data(text, tokenizer):
    tokens = tokenizer(text, max_length=256, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='tf')
    return ({'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']})

def mBertClassifier(text):
    path = "C:\\Users\\Golnoosh\\Desktop\\University\\Sem8\\Final Project\\Final-Models\\GhazalRobaeeModels\\GhazalRobaeemBERT\\GhazalRobaeemBERT\\weights"
    SEQ_LEN = 256
    global normalizer 
    normalizer = hazm.Normalizer()
    label2id = {'abusaeed':0, 'attar':1, 'hafez':2, 'moulavi':3, 'saadi':4, 'sanaee':5}
    id2label = {0: 'abusaeed', 1: 'attar', 2: 'hafez', 3: 'moulavi', 4: 'saadi', 5: 'sanaee'}

    MODEL_NAME = 'bert-base-multilingual-cased'
    config = BertConfig.from_pretrained(
    MODEL_NAME, **{
        'label2id': label2id,
        'id2label': id2label,
    })
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    base_model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')
    classificationResult = base_model([input_ids, mask]).logits
    Final_output = tf.keras.layers.Dense(5, activation ='softmax', trainable=True)(classificationResult)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=Final_output)

    model.load_weights(path).expect_partial()

    df1 = pd.DataFrame(columns=['poem', 'poet'])
    df1.loc[0] = [text, "saadi"]
    id_label_map = {
       'abusaeed': 0
        , 'attar': 1
        , 'hafez': 2
        , 'moulavi':3
        , 'saadi': 4
        , 'sanaee':5
    }
    df1['poet'] = df1['poet'].map(id_label_map)

    df1['cleaned_poem'] = df1['poem'].apply(cleaning)
    df1 = df1[['cleaned_poem', 'poet']]
    df1.columns = ['poem', 'poet']
    

    df1['cleaned_poem'] = df1['poem'].apply(removeStopWords)
    df1 = df1[['cleaned_poem', 'poet']]
    df1.columns = ['poem', 'poet']

    df1['predicted-label'] = None
    tokens = prep_data(df1['poem'].tolist(), tokenizer)
    prb = model.predict(tokens)
    probabilty = '{:0.2f}'.format(max(prb[0]))
    pred_result = np.argmax(prb)
    df1['predicted-label'] = pred_result

    idToLabel = [ 'abusaeed'
        , 'attar'
        , 'hafez'
        , 'moulavi'
        , 'saadi'
        , 'sanaee']


    reslist = []
    predictres = idToLabel[pred_result]
    reslist.append(predictres)
    reslist.append(probabilty)
    return reslist

   





