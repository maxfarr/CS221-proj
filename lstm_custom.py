import collections
import os
import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Bidirectional
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import random
import numpy as np
import pandas as pd
import argparse

data_file = "popular_quotes_clean.csv"
custom = True
BATCH_SIZE = 20
HIDDEN_DIM = 20

def embed(df):
    file = "glove.6B.100d.txt"
    with open(file, 'r', encoding="utf-8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    words = [x.split() for x in df["text"].astype(str).tolist()]
    words = set([x for sublist in words for x in sublist] + ["<bos>", "<eos>"])
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}
    print("intook embedding")
    vector_dim = len(vectors["the"])
    W = np.zeros((vocab_size, vector_dim))
    for i in range(vocab_size):
        if ivocab[i] in vectors:
            W[i] = vectors[ivocab[i]]
        else:
            W[i] = vectors["<unk>"]
            vocab[i] = "<unk>"
            ivocab["<unk>"] = i
    print("created embedding matrix")

    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return W_norm, vocab, ivocab, vocab_size, vector_dim

df = pd.read_csv(data_file)
embedding, word_to_id, id_to_word, vocab_len, vector_dim = embed(df)

text = df["text"].astype(str).tolist()
text_to_ids = [[word_to_id["<bos>"]] + [word_to_id[word] for word in sentence.split()] + [word_to_id["<eos>"]]\
    for sentence in text]
likes = df["likes"].astype(int).tolist()
likes = [like / sum(likes) for like in likes] * 1000
ids_to_likes = dict(zip([tuple(lis) for lis in text_to_ids], likes))
maxlen = 32
text_to_ids = [sentence for sentence in text_to_ids if len(sentence) < maxlen]
#maxlen = max([len(sentence) for sentence in text_to_ids])

#create_input and output
x = np.zeros((len(text_to_ids), maxlen, vector_dim))
y = np.zeros((len(text_to_ids), maxlen, vocab_len)) #m by vocab_len
for i, sentence in enumerate(text_to_ids):
    for t, word_id in enumerate(sentence):
        x[i , t] = embedding[word_id]
        if(word_id != word_to_id["<eos>"]):
            y[i, t, sentence[t+1]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Bidirectional(LSTM(maxlen, return_sequences=True)))
#model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(vector_dim)))
model.add(Dense(vocab_len, activation='softmax'))
#model.add(Activation('softmax'))

def custom_loss(target, output):
    def get_index(one_hot):
        try:
            return one_hot.index(1)
        except ValueError:
            return -1
    target_ids = tf.map_fn(get_index, target)
    target_ids = [one_hot.index(1) for one_hot in target_list[1:]]
    target_ids = [i for i in target_ids if i >= 0]
    num_likes = ids_to_likes[tuple(target_ids)]
    return K.categorical_crossentropy(target, output) * num_likes

optimizer = Adam(lr=0.01)
if custom:
    model.compile(loss=custom_loss, optimizer=optimizer)
else:
    model.compile(loss="categorical_crossentropy", optimizer = optimizer)


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
    sentence = "<bos>"
    
    for i in range(5):
        round = 1
        while sentence.split()[-1] != "<eos>" and round < maxlen-3:
            ind_sentence = [word_to_id[word] for word in sentence.split()]
            x_pred = np.zeros((1, maxlen, vector_dim))
            for t, word in enumerate(ind_sentence):
                x_pred[0, t] = embedding[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds[round])
            next_word = id_to_word[next_index]

            sentence = sentence + " " + next_word
            round += 1
        print(sentence)

        
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(x, y,
          batch_size=BATCH_SIZE,
          epochs=20,
          callbacks=[print_callback])
model.save("popular_quotes_embed.hdf5")