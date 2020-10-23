import io
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from functools import partial
import os

from sys import stdout


from numpy.random import randint, rand, randn
from numpy import ones, zeros

import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

logging_format = '%(asctime)s %(levelname)s: %(message)s'
logging_file = 'training_log.txt'
logging.basicConfig(level=logging.INFO, format=logging_format,
                        handlers=[logging.StreamHandler(stdout), logging.FileHandler(logging_file, mode='w')])

train_path = 'train.txt'
text = []
with open(train_path, 'r') as f:
    for line in f:
        words = line[:-1].replace('.', '').lower()
        text.append(words)


token = Tokenizer(char_level=True)
token.fit_on_texts(text)

word_index = token.word_index
index_word = {v: k for k, v in word_index.items()}

converted = token.texts_to_sequences(text)
padded = pad_sequences(converted, maxlen=32, truncating='post', padding='post')

print([index_word[n] for n in padded[100]])


src_txt_length = 32
vocab_size = len(word_index)
sum_txt_length = 32

inputs = Input(shape=(src_txt_length,))
encoder1 = Embedding(vocab_size, 128)(inputs)
encoder2 = LSTM(128)(encoder1)
encoder3 = RepeatVector(sum_txt_length)(encoder2)
decoder1 = LSTM(128, return_sequences=True)(encoder3)
outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)
ae = Model(inputs=inputs, outputs=outputs)
ae.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

print(ae.summary())


in_shape=(src_txt_length, vocab_size)
disc_in = Input(shape=in_shape)
disc_model = Conv1D(1024, 3, padding='same', input_shape=in_shape, activation='relu')(disc_in)
disc_model = Dropout(0.4)(disc_model)
disc_model = Conv1D(1024, 3, padding='same', activation='relu')(disc_model)
disc_model = Dropout(0.4)(disc_model)
disc_model = Flatten()(disc_model)
disc_model = Dense(1, activation='sigmoid')(disc_model)
disc = Model(inputs=disc_in, outputs=disc_model)
disc.compile(loss='binary_crossentropy', optimizer='adam')
print(disc.summary())


latent_dim = 500

# Sequential model used here as functional interface was having issue for some reason
gen = Sequential()
n_nodes = 32*57
gen.add(Dense(n_nodes, activation='relu', input_shape=(latent_dim,)))
gen.add(Reshape((32,57)))
gen.add(LSTM(57,return_sequences=True))
gen.add(Dropout(0.2))
gen.add(LSTM(57, return_sequences=True))
gen.add(Dropout(0.2))
gen.add(LSTM(57, return_sequences=True))
gen.add(Dropout(0.2))
gen.add(LSTM(57, return_sequences=True))
gen.add(Dropout(0.2))
gen.add(LSTM(57, return_sequences=True))
gen.add(Dropout(0.2))
gen.add(LSTM(57, return_sequences=True))

print(gen.summary())

# Gan definition and structure taken from https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
def define_gan(g_model, d_model):
  d_model.trainable = False
  model = Sequential()
  model.add(g_model)
  model.add(d_model)
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model

gan = define_gan(gen, disc)
print(gan.summary())


def generate_real_samples(dataset, n_samples=32):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	while len(word_index) in X:
		ix = randint(0, dataset.shape[0], n_samples)
		X = dataset[ix]
	y = ones((n_samples, 1))
	return X, y

def generate_latent_points(latent_dim=500, n_samples=32):
	x_input = randn(latent_dim * n_samples)
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

def generate_fake_samples(g_model, latent_dim=500, n_samples=32):
	x_input = generate_latent_points(latent_dim, n_samples)
	X = g_model.predict(x_input)
	y = zeros((n_samples, 1))
	return X, y


def train(ae, disc, gen, gan, dataset, n_iter=200000, n_batch=64):
	for i in range(n_iter):
		X_real, y_real = generate_real_samples(dataset, n_batch)
		ae_loss = ae.train_on_batch(X_real, X_real)

		if ((i+1)%50) == 0:
			logging.info('{} ae_loss={}'.format(i+1, ae_loss))

		X_real, y_real = generate_real_samples(dataset, n_batch)
		X_distilled = ae.predict(X_real)

		X_fake, y_fake = generate_fake_samples(gen, n_samples=n_batch)

		X_full = np.concatenate((X_distilled, X_fake))
		y_full = np.concatenate((y_real, y_fake))

		disc_loss = disc.train_on_batch(X_full, y_full)

		if ((i+1)%50) == 0:
			logging.info('{} disc_loss={}'.format(i+1, disc_loss))

		for _ in range(5):
			X_gan = generate_latent_points(latent_dim, n_batch)
			y_gan = ones((n_batch, 1))
			gan_loss = gan.train_on_batch(X_gan, y_gan)

		if ((i+1)%50) == 0:
			logging.info('{} Gan_loss={}'.format(i+1, gan_loss))
			gen.save('GeneratorModel')
			logging.info('Saved model')


train(ae, disc, gen, gan, padded)