from tensorflow.keras.models import load_model
from numpy.random import randn
import os
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

model = load_model('GeneratorModel')

latent_dim = 500
n_samples = 5

x_input = randn(latent_dim * n_samples)
x_input = x_input.reshape(n_samples, latent_dim)

predictions = model.predict(x_input)

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
index_word[0] = ''

for pred in predictions:
	sent = ''
	for ch in pred:
		sent += index_word[np.argmax(ch)]
	print(sent)

print(index_word)

print(model.summary())