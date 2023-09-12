import csv
import glob
import pandas as pd
import os
import numpy as np
import hazm
import re
from hazm import stopwords_list
from transformers import AutoConfig, AutoTokenizer
from transformers import BertConfig, BertTokenizer
from transformers import TFXLMRobertaForSequenceClassification
import tensorflow as tf




def prep_data(text, tokenizer):
    tokens = tokenizer(text, max_length=256, truncation=True, padding='max_length', add_special_tokens=True, return_tensors='tf')
    return ({'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']})

def RobaeeRobertaClassifier(text):
    path = "C:\\Users\\Golnoosh\\Desktop\\University\\Sem8\\Final Project\\Final-Models\\Robaee\\RobaeeRobertaModel\\RobaeeRobertaModel\\weights"

    SEQ_LEN = 256
    BATCH_SIZE = 20
    label2id =  {'abusaeed': 0, 'attar': 1, 'moulavi': 2}
    id2label = {0: 'abusaeed', 1: 'attar', 2: 'moulavi'}
    MODEL_NAME = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = TFXLMRobertaForSequenceClassification.from_pretrained(
                                                                        "xlm-roberta-base",
                                                                        num_labels = 3,
                                                                        id2label=id2label,
                                                                        label2id=label2id,
                                                                        )
    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')
    classificationResult = base_model([input_ids, mask]).logits
    Final_output = tf.keras.layers.Dense(3, activation ='softmax', trainable=True)(classificationResult)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=Final_output)

    model.load_weights(path).expect_partial()

    df1 = pd.DataFrame(columns=['poem', 'poet'])
    df1.loc[0] = [text, "saadi"]
    id_label_map = {
       'abusaeed': 0, 'attar': 1, 'moulavi': 2
    }
    df1['poet'] = df1['poet'].map(id_label_map)

    df1['predicted-label'] = None
    tokens = prep_data(df1['poem'].tolist(), tokenizer)
    prb = model.predict(tokens)
    probabilty = '{:0.2f}'.format(max(prb[0]))

    pred_result = np.argmax(prb)
    df1['predicted-label'] = pred_result

    idToLabel = [ 'abusaeed', 'attar', 'moulavi']


    reslist = []
    predictres = idToLabel[pred_result]
    reslist.append(predictres)
    reslist.append(probabilty)
