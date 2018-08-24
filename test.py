# # CODE

# First, lets import all the necessary packages

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #USE GPU
from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords


from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta

from property import Property
from property import load_property

# Global variables
THRESHOD = 0.5

# File paths
QUESTIONS_CSV = 'questions.csv'
TEST_CSV = 'test.csv'
EMBEDDING_FILE = '/data/caohao/GoogleNews-vectors-negative300.bin'
#EMBEDDING_FILE = '/data/caohao/wiki-news-300d-1M.vec'
MODEL_SAVING_DIR = 'models/'
MODEL_NAME = 'weights.h5'
PROPERTY_PATH = 'property.pkl'

# Create embedding matrix

# Load test set
questions_df = pd.read_csv(QUESTIONS_CSV)
test_df = pd.read_csv(TEST_CSV)

# Load property
p = load_property(PROPERTY_PATH)
print("max_seq_length: " + str(p.max_seq_length))

stops = set(stopwords.words('english'))

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

# Prepare embedding
print("Loading word vectors...")
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)


questions_col = 'title'
# Iterate over the questions of training datasets
print("Word encoding...")
for dataset in [questions_df, test_df]:
    for index, row in dataset.iterrows():
        q2n = []  # q2n -> question numbers representation
        for word in text_to_word_list(row[questions_col]):

            # Check for unwanted words
            # if word in stops and word not in word2vec.vocab:
            #     continue
            if word in stops:
                continue

            if word not in p.vocabulary:
                continue

            q2n.append(p.vocabulary[word])

            if len(q2n) > p.max_seq_length:
                break

        # Replace questions as word to question as number representation
        dataset.set_value(index, questions_col, q2n)

embedding_dim = 300
embeddings = 1 * np.random.randn(len(p.vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
print("Building embedding matrix...")
for word, index in p.vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec

# Prepare for test
X_left = questions_df.title
test_set = test_df.title

# Zero padding
X_left = pad_sequences(X_left, maxlen=p.max_seq_length)
test_set = pad_sequences(test_set, maxlen=p.max_seq_length)

# Build the model

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25


def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


# The visible layer
left_input = Input(shape=(p.max_seq_length,), dtype='int32')
right_input = Input(shape=(p.max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=p.max_seq_length,
                            trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])


# load model
print('Loading model...')
malstm.load_weights(MODEL_NAME, by_name=True)


# predict
print("Predicting...")
questions_num = len(X_left)
for i, row in enumerate(test_set):
    # tile [1, max_seq_length] row array to [questions_num, max_seq_length] array
    X_right = np.tile(row, (questions_num, 1))
    preds = malstm.predict([X_left, X_right])
    preds = preds.flatten()
    max_index = np.argmax(preds)
    if preds[max_index] >= THRESHOD:
        print(str(i+1) + " test question, duplicate question id: " + str(questions_df['qid'][max_index]) + ", duplicate_prob: " + str(preds[max_index]))
    else:
        print(str(i+1) + " test question, cannot find duplicate question!")

