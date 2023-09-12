import pandas as pd
import hazm
import re
import tensorflow as tf
import keras
from hazm import stopwords_list
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences





def LSTMClassifier(text):
    path = "C:\\Users\\Golnoosh\\Desktop\\University\\Sem8\\Final Project\\Final-Models\\Ghazal\\GhazalLSTMModel\\GhazalLSTMModel"

    MAX_NB_WORDS = 10000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 256
    # This is fixed.
    EMBEDDING_DIM = 100

    loaded_model = keras.models.load_model(path, compile=False)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')

    df1 = pd.DataFrame(columns=['poem', 'poet'])
    df1.loc[0] = [text, "hafez"]
    

    X_new = tokenizer.texts_to_sequences(df1['poem'].values)
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = loaded_model.predict(X_new)
    list =  prediction.tolist()
    prb = '{:0.2f}'.format(max(prediction[0]))
    Y_new_pred= prediction.argmax(axis=-1).tolist()
    Y_new_pred

    idToLabel = [ "attar",
        "hafez",
        "moulavi",
        "saadi",
        "sanaee"]

    reslist = []
    poet = idToLabel[Y_new_pred[0]]
    reslist.append(poet)
    reslist.append(prb)
    
    return reslist

