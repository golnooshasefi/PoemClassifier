import tensorflow as tf
import pandas as pd
# import matplotlib.pyplot as plt
import hazm
import re
import csv
import glob
import os
import numpy as np
from hazm import stopwords_list
from transformers import AutoConfig, AutoTokenizer
from transformers import TFAutoModel, TFBertForSequenceClassification



def prep_data(text, tokenizer):
    tokens = tokenizer(text, max_length=256, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='tf')
    return ({'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']})

def ParsBertClassifier(text):
    path = "C:\\Users\\Golnoosh\\Desktop\\University\\Sem8\\Final Project\\Final-Models\\Ghazal\\GhazalParsbertModel\\GhazalParsbertModel\\weights"
    SEQ_LEN = 256
    label2id = {'attar': 0, 'hafez': 1, 'moulavi': 2, 'saadi': 3, 'sanaee': 4}
    id2label = {0: 'attar', 1: 'hafez', 2: 'moulavi', 3: 'saadi', 4: 'sanaee'}

    MODEL_NAME = 'HooshvareLab/bert-fa-base-uncased'
    config = AutoConfig.from_pretrained(
        MODEL_NAME, **{
            'label2id': label2id,
            'id2label': id2label,
        })
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')
    modelResult = base_model([input_ids, mask]).logits
    new_Final_output = tf.keras.layers.Dense(5, activation ='softmax', trainable=True)(modelResult)
    new_model = tf.keras.Model(inputs=[input_ids, mask], outputs=new_Final_output)

    new_model.load_weights(path).expect_partial()
    df1 = pd.DataFrame(columns=['poem', 'poet'])
    df1.loc[0] = [text, "saadi"]
    id_label_map = {
        "attar": 0,
        "hafez": 1,
        "moulavi": 2,
        "saadi": 3,
        "sanaee": 4
    }
    df1['poet'] = df1['poet'].map(id_label_map)

    # df1['cleaned_poem'] = df1['poem'].apply(cleaning)
    # df1 = df1[['cleaned_poem', 'poet']]
    # df1.columns = ['poem', 'poet']
    

    # df1['cleaned_poem'] = df1['poem'].apply(removeStopWords)
    # df1 = df1[['cleaned_poem', 'poet']]
    # df1.columns = ['poem', 'poet']

    
    df1['predicted-label'] = None
    tokens = prep_data(df1['poem'].tolist(), tokenizer)
    prb = new_model.predict(tokens)
    probabilty = '{:0.2f}'.format(max(prb[0]))
    pred_result = np.argmax(prb)
    df1['predicted-label'] = pred_result

    idToLabel = [ "attar",
    "hafez",
    "moulavi",
    "saadi",
    "sanaee"]

    reslist = []
    predictres = idToLabel[pred_result]
    reslist.append(predictres)
    reslist.append(probabilty)

    


    return reslist
    
