import pandas as pd
import hazm
import re
import tensorflow as tf
import keras
from hazm import stopwords_list
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

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

def Genreclassifier(text):
    path = "C:\\Users\\Golnoosh\\Desktop\\University\\Sem8\\Final Project\\Final-Models\\GenreLSTMModel\\GenreLSTMModel"
    global normalizer 
    normalizer = hazm.Normalizer()
    loaded_model = keras.models.load_model(path)
    df1 = pd.DataFrame(columns=['poem', 'genre'])
    df1.loc[0] = [text, "robaee"]
    id_label_map = {
    "ghazal": 0,
    "robaee": 1,
    }
    df1['genre'] = df1['genre'].map(id_label_map)
    df1['cleaned_poem'] = df1['poem'].apply(cleaning)
    df1 = df1[['cleaned_poem', 'genre']]
    df1.columns = ['poem', 'genre']

    df1['genre'] = df1['genre'].map(id_label_map)
    df1['cleaned_poem'] = df1['poem'].apply(removeStopWords)
    df1 = df1[['cleaned_poem', 'genre']]
    df1.columns = ['poem', 'genre']

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 50000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 256
    # This is fixed.
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
    tokenizer.fit_on_texts(df1['poem'].values)
    X_new = tokenizer.texts_to_sequences(df1['poem'].values)
    X_new = pad_sequences(X_new, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = loaded_model.predict(X_new)
    Y_new_pred= prediction.argmax(axis=-1).tolist() 
    idToLabel = [ "ghazal", "robaee"]

    result = idToLabel[Y_new_pred[0]] 
    return result

# if __name__=="__main__":
#     # normalizer = hazm.Normalizer()
#     path = "./GenreLSTMModel/GenreLSTMModel"
#     str = "چه کند بنده که گردن ننهد فرمان را چه کند گوی که عاجز نشود چوگان را سروبالای کمان ابرو اگر تیر زند عاشق آنست که بر دیده نهد پیکان را دست من گیر که بیچارگی از حد بگذشت سر من دار که در پای تو ریزم جان را کاشکی پرده برافتادی از آن منظر حسن تا همه خلق ببینند نگارستان را همه را دیده در اوصاف تو حیران ماندی تا دگر عیب نگویند من حیران را لیکن آن نقش که در روی تو من می‌بینم همه را دیده نباشد که ببینند آن را چشم گریان مرا حال بگفتم به طبیب گفت یک بار ببوس آن دهن خندان را گفتم آیا که در این درد بخواهم مردن که محالست که حاصل کنم این درمان را پنجه با ساعد سیمین نه به عقل افکندم غایت جهل بود مشت زدن سندان را سعدی از سرزنش خلق نترسد هیهات غرقه در نیل چه اندیشه کند باران را سر بنه گر سر میدان ارادت داری ناگزیرست که گویی بود این میدان را"
#     pred =   Genreclassifier(str, path)
#     print(pred)
