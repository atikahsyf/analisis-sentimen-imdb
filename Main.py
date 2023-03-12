import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model   # load saved model
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from sklearn.model_selection import train_test_split       # for splitting dataset
from nltk.corpus import stopwords   # to get collection of stopwordsnote
import pandas as pd    # to load dataset
import numpy as np     # for mathematic equation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
# nltk.download('punkt')
# to do padding or truncating
# layers of the architecture

#
data = pd.read_csv('IMDB Dataset.csv')
dataTes = pd.read_csv('dataset_test.csv')
# print(data)

#
english_stops = set(stopwords.words('english'))

#


def load_dataset():
    df = pd.read_csv('IMDB Dataset.csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    # remove html tag
    x_data = x_data.replace({'<.*?>': ''}, regex=True)
    # remove non alphabet
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex=True)
    x_data = x_data.apply(lambda review: [w for w in review.split(
    ) if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower()
                          for w in review])   # lower case

    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    x_data = x_data.map(' '.join)

    return x_data, y_data


def load_dataset_test():
    df = pd.read_csv('dataset_test.csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    # remove html tag
    x_data = x_data.replace({'<.*?>': ''}, regex=True)
    # remove non alphabet
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex=True)
    x_data = x_data.apply(lambda review: [w for w in review.split(
    ) if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower()
                          for w in review])   # lower case

    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data


x_data, y_data = load_dataset()
x_data_test, y_data_test = load_dataset_test()


#
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)

#


def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))


# ENCODE REVIEW
# no need lower, because already lowered the data in load_data()

pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                    ('lr_clf', LogisticRegression())])
pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

token = Tokenizer(lower=False)
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)
x_data_test = token.texts_to_sequences(x_data_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length,
                        padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length,
                       padding='post', truncating='post')
x_data_test = pad_sequences(x_data_test, maxlen=max_length,
                            padding='post', truncating='post')


total_words = len(x_train) + 1   # add 1 because of 0 padding

# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64
lr = 1e-2
epoch = 10
batch_size = 64

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(LSTM_OUT, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=lr, decay=lr)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy'])


# TRAINING

def train_data():
    h = model.fit(x_train, y_train, batch_size=batch_size,
                  validation_data=(x_test, y_test), epochs=epoch)

    # TESTING
    model.predict(x_test, batch_size=batch_size)
    # y_predict = np.argmax(model.predict(x_test, batch_size = 128), axis=1) # kalau make softmax

    model.save('models/LSTMv1.h5')

    plt.style.use("ggplot")
    plt.figure()
    N = epoch
    plt.plot(np.arange(0, N), h.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), h.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), h.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), h.history["val_accuracy"], label="val_accuracy")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/accuracy")
    plt.legend(loc="upper right")

    # save plot
    filename = f'plot-{epoch}-{batch_size}.png'
    plt.savefig(filename)

    acc = h.history["val_accuracy"]
    # print(acc.last_index())

    # TESTING v2 using made up dataset
    print("\nPredict using made up dataset\n")
    y_predict = (model.predict(x_data_test, batch_size=64) >
                 0.5).astype("int32")  # kalau make sigmoid

    true = 0
    for i, y in enumerate(y_data_test):
        if y == y_predict[i]:
            true += 1

    print('Correct Prediction: {}'.format(true))
    print('Wrong Prediction: {}'.format(len(y_predict) - true))
    print('Accuracy: {}'.format(true/len(y_predict)*100))

    return len(acc)-1


def predict(review):
    # LOAD SAVE MODEL
    loaded_model = load_model('models/LSTMv1.h5')

    #review = 'Despite a good theme, great acting and important messages that this movie convey in an unorthodox way, I think it fails to connect the audience with the storyline and leaves him in a world of confusion. Although, majority of reviews find this movie entertaining and interesting, yet I would choose to be a minority that believes that this movie is extremely overrated.'

    x_testing = review
    # PRE-PROCESS REVIEW
    # remove html tag
    x_testing = x_testing.replace({'<.*?>': ''}, regex=True)
    # remove non alphabet
    x_testing = x_testing.replace({'[^A-Za-z]': ' '}, regex=True)
    x_testing = x_testing.apply(lambda review: [w for w in review.split(
    ) if w not in english_stops])  # remove stop words
    x_testing = x_testing.apply(lambda review: [w.lower()
                                                for w in review])   # lower case

    x_testing = x_testing.map(' '.join)

    #
    tokenize_words = token.texts_to_sequences(x_testing)
    tokenize_words = pad_sequences(
        tokenize_words, maxlen=max_length, padding='post', truncating='post')

    #
    result = loaded_model.predict(tokenize_words)

    if y >= 0.7:
        res = "positive"
    else:
        res = "negative"
    return res
