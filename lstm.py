import collections
import os
import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import random
import numpy as np
import pandas as pd
import argparse

data_file = "popular_quotes_clean.csv"
BATCH_SIZE = 40
HIDDEN_DIM = 60
#h
def build_vocab(df):
    corpus = [x.split() for x in df["text"].astype(str).tolist()]
    corpus = set([x for sublist in corpus for x in sublist] + ["<eos>"])
    word_to_id = dict(zip(corpus, range(len(corpus))))
    return len(corpus) + 1, word_to_id

df = pd.read_csv(data_file)
vocab_len, word_to_id = build_vocab(df)
id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

text = df["text"].astype(str).tolist()
text_to_ids = [[word_to_id[word] for word in sentence.split()] + [word_to_id["<eos>"]] for sentence in text]
maxlen = max([len(sentence) for sentence in text_to_ids])

#so that I don't get a memory error
batch_size = 20
#batch = text_to_ids[0:batch_size]
batch = text_to_ids
#create_input
x = np.zeros((len(batch), maxlen, vocab_len), dtype=np.bool)
y = np.zeros((len(batch), maxlen, vocab_len), dtype=np.bool)
for i, sentence in enumerate(batch):
    for t, word in enumerate(sentence):
        x[i, t, word] = 1
        if(t < len(sentence)):
            y[i, t, sentence[t]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(Dense(vocab_len, activation='softmax'))

optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    start_index = random.randint(0, vocab_len - 1)
    sentence = id_to_word[start_index]
    print('----- Generating with seed: "' + sentence + '"')

    for i in range(10):
        ind_sentence = [word_to_id[word] for word in sentence.split()]
        x_pred = np.zeros((1, maxlen, vocab_len))
        for t, word in enumerate(ind_sentence):
            x_pred[0, t, word] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds[i+1])
        next_word = id_to_word[next_index]

        sentence = sentence + " " + next_word
    print(sentence)
        
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y,
          batch_size=BATCH_SIZE,
          epochs=5,
          callbacks=[print_callback])
model.save("popular_quotes_clean_model.hdf5")
