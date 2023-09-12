import pandas as pd
import hazm
import re
import tensorflow as tf
import keras
from hazm import stopwords_list
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# def stop_word_importer(file_name):# importing persian stopwords
#     with open(file_name, 'r', encoding="utf8") as myfile:
#         stop_words = myfile.read().replace('\n', ' ').replace("\u200c","").replace("\ufeff","").replace("."," ").split(' ')# a list of stop words
#     return stop_words

# def removeStopWords(text):
#     stop_words = stop_word_importer('./stop_words.txt') 
#     text = ' '.join([word for word in text.split() if word not in stopwords_list()])
#     text = ' '.join([word for word in text.split() if word not in stop_words])
#     return text

# def cleaning(text):
#     text = text.strip()
#     # normalizing
#     text = normalizer.normalize(text)

#     # replacing all spaces,hyphens,  with white space
#     space_pattern = r"[\xad\ufeff\u200e\u200d\u200b\x7f\u202a\u2003\xa0\u206e\u200c\x9d\]]"
#     space_pattern = re.compile(space_pattern)
#     text = space_pattern.sub(" ", text)

#     # let's delete the un-required elements
#     deleted_pattern = r"(\d|[\|\[]]|\"|'ٍ|[0-9]|¬|[a-zA-Z]|[؛“،,”‘۔’’‘–]|[|\.÷+\:\-\?»\=\{}\*«_…\؟!/ـ]|[۲۹۱۷۸۵۶۴۴۳]|[\\u\\x]|[\(\)]|[۰'ٓ۫'ٔ]|[ٓٔ]|[ًٌٍْﹼ،َُِّ«ٰ»ٖء]|\[]|\[\])"
#     deleted_pattern = re.compile(deleted_pattern)
#     text = deleted_pattern.sub("", text).strip()


#     # removing wierd patterns
#     wierd_pattern = re.compile("["
#         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#         u"\U0001F680-\U0001F6FF"  # transport & map symbols
#         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         u"\U00002702-\U000027B0"
#         u"\U000024C2-\U0001F251"
#         u"\U0001f926-\U0001f937"
#         u'\U00010000-\U0010ffff'
#         # u"\0x06F0-\0x06F9"
#         u"\u200d"
#         u"\u200c"
#         u"\u2640-\u2642"
#         u"\u2600-\u2B55"
#         u"\u23cf"
#         u"\u23e9"
#         u"\u231a"
#         u"\u3030"
#         u"\ufe0f"
#         u"\u2069"
#         u"\u2066"
#         u"\u2068"
#         u"\u2067"
#         "]+", flags=re.UNICODE)

#     text = wierd_pattern.sub(r'', text)
#     # removing extra spaces, hashtags
#     text = re.sub("#", "", text)
#     text = re.sub("\s+", " ", text)
#     return text

def GRUClassifier(text):
    path = "C:\\Users\\Golnoosh\\Desktop\\University\\Sem8\\Final Project\\Final-Models\\GhazalRobaeeModels\\GhazalRobaeeGRU\\GhazalRobaeeGRU"  
    MAX_NB_WORDS = 10000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 256
    # This is fixed.
    EMBEDDING_DIM = 100
    loaded_model = keras.models.load_model(path, compile=False)
    df1 = pd.DataFrame(columns=['poem', 'poet'])
    df1.loc[0] = [text, "hafez"]
    id_label_map = {
                    "abusaeed": 0,
                    "attar": 1,
                    "hafez": 2,
                    "moulavi": 3,
                    "saadi": 4,
                    "sanaee": 5
                    }
    df1['poet'] = df1['poet'].map(id_label_map)
    

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    tokenizer.fit_on_texts(df1['poem'].values) 
    X_new = tokenizer.texts_to_sequences(df1['poem'].values)
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH) 
    prediction = loaded_model.predict(X_new)
    list =  prediction.tolist()
    prb = '{:0.2f}'.format(max(prediction[0]))
    Y_new_pred= prediction.argmax(axis=-1).tolist()
    idToLabel = [ 'abusaeed'
        , 'attar'
        , 'hafez'
        , 'moulavi'
        , 'saadi'
        , 'sanaee']

    reslist = []
    poet = idToLabel[Y_new_pred[0]]
    reslist.append(poet)
    reslist.append(prb)
    
    return reslist


