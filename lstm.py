import collections
import os
import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Bidirectional
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import random
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

data_file = "popular_quotes_clean.csv"
custom = False
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
    vector_dim = len(vectors[ivocab[0]])
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
maxlen = 32
text_to_ids = [sentence for sentence in text_to_ids if len(sentence) < maxlen]
#maxlen = max([len(sentence) for sentence in text_to_ids])
likes = df["likes"].astype(int).tolist()
likes = [like / sum(likes) for like in likes]
#create_input and output
x = np.zeros((len(text_to_ids), maxlen, vector_dim))
if custom:
    y = np.zeros((len(text_to_ids), maxlen, vocab_len + 1)) #m by vocab_len
else:
    y = np.zeros((len(text_to_ids), maxlen, vocab_len)) #m by vocab_len
for i, sentence in enumerate(text_to_ids):
    for t, word_id in enumerate(sentence):
        x[i , t] = embedding[word_id]
        if(word_id != word_to_id["<eos>"]):
            y[i, t, sentence[t+1]] = 1
            if custom:
                y[i, t, -1] = likes[i]

def build_lstm(learning_rate=0.01, b_1=0.9, b_2=0.999):
    mod = Sequential()
    mod.add(Bidirectional(LSTM(maxlen, return_sequences=True)))
    #model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    mod.add(TimeDistributed(Dense(vector_dim)))
    if custom:    
        mod.add(Dense(vocab_len+1, activation='softmax'))
    else:
        mod.add(Dense(vocab_len, activation='relu'))
    mod.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate, beta_1=b_1, beta_2=b_2)
    if custom:
        mod.compile(loss=custom_loss(), optimizer=optimizer)
    else:
        mod.compile(loss="categorical_crossentropy", optimizer = optimizer)
    return mod

p_grid = {
    "learning_rate" : [0.001, 0.01, 0.1, 0.2],
    "b_1" : [0.6, 0.75, 0.9],
    "b_2" : [0.7, 0.8, 0.999]
    #"adjust_likes" : [1, 10, 100, 1000, 10000]
}

# build the model: a single LSTM
print('Build model...')
model = GridSearchCV(KerasClassifier(build_fn = build_lstm), param_grid=p_grid)

def custom_loss(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.
    # Returns
        Output tensor.
    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    likes = target[-1]
    target = target[:-1]
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return -(tf.reduce_sum(target * tf.log(output), axis) * likes * adjust_likes)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


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
    
    for i in range(5):
        sentence = "<bos>"
        round = 1
        while sentence.split()[-1] != "<eos>" and round < maxlen-3:
            ind_sentence = [word_to_id[word] for word in sentence.split()]
            x_pred = np.zeros((1, maxlen, vector_dim))
            for t, word in enumerate(ind_sentence):
                x_pred[0, t] = embedding[word]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds[round-1])
            next_word = id_to_word[next_index]

            sentence = sentence + " " + next_word
            round += 1
        ind_sentence = [word_to_id[word] for word in sentence.split()]
        for i in range(1, len(ind_sentence) - 2, -1):
            x_pred = np.zeros((1, maxlen, vector_dim))
            for t, word in enumerate(ind_sentence):
                x_pred[0, t] = embedding[word]

            preds = model.predict(x_pred, verbose=0)[0]
            word_id = sample(preds[i])
            ind_sentence[i+1] = word_id
        sentence = " ".join([id_to_word[id] for id in ind_sentence])
        print(sentence)

        
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
es = EarlyStopping(monitor='val_loss')
model.fit(x, y,
          batch_size=BATCH_SIZE,
          epochs=20,
          callbacks=[print_callback, es])
model.save("popular_quotes_grid.hdf5")
