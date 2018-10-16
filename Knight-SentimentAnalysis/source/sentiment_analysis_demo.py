# -*- coding: utf-8 -*-

# imports
from re import sub
from collections import Counter
import tensorflow as tf
from tensorflow import keras

# hyperparameters and constants
max_vocab_size = 5000
num_embedding_dims = 128
num_hidden_nodes = 64
sequence_max_len = 100
batch_size = 150
epochs = 20
UNKNOWN_WORD = -1

# list of files of lines of tab-separated values
text_dataset_filenames = [
        '..\\data\\sentiment labelled sentences\\amazon_cells_labelled.txt',
        '..\\data\\sentiment labelled sentences\\imdb_labelled.txt',
        '..\\data\\sentiment labelled sentences\\yelp_labelled.txt'
        ]

# extract data and labels from files
data = []
labels = []
for filename in text_dataset_filenames:
    with open(filename) as file:
        for line in file:
            data.append(line[ : -3])
            labels.append(int(line[-2]))

# clean up data
def clean_sentence(sentence):
    return sub('[^0-9a-z ]+', '', sentence.lower())

for i in range(len(data)):
    data[i] = clean_sentence(data[i])

# get sorted (most frequent first) vocabulary
all_words = []
for line in data:
    for word in line.split():
        all_words.append(word)
word_frequency = Counter(all_words)
vocab_tuples_sorted = word_frequency.most_common()
vocab = [vocab_tuple[0] for vocab_tuple in vocab_tuples_sorted]
vocab = vocab[ : max_vocab_size - 1]
vocab.append('unknownword')

# make numeric ID encoder
def encode_text_to_IDs(text):
    words = text.split()
    ids = []
    for word in words:
        try:
            ids.append(vocab.index(word))
        except ValueError:
            ids.append(UNKNOWN_WORD)
    return ids

# make numeric ID decoder
def decode_IDs_to_text(IDs):
    return ' '.join([vocab[ID] for ID in IDs])

# make model
model = keras.Sequential()
model.add(keras.layers.Embedding(max_vocab_size, num_embedding_dims))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(num_hidden_nodes, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# configure for training
model.compile(tf.train.AdamOptimizer(), 'binary_crossentropy', ['accuracy'])

# produce training_data and training_labels
#     encode and pad data
training_data = []
for datum in data:
    training_data.append(encode_text_to_IDs(datum))
training_data = keras.preprocessing.sequence.pad_sequences(training_data, sequence_max_len, padding='post', value=UNKNOWN_WORD)
training_labels = labels

# train model
model.fit(training_data, training_labels, batch_size, epochs)

# apply net to new input
def get_sentiment(sentence):
    return model.predict(keras.preprocessing.sequence.pad_sequences([encode_text_to_IDs(clean_sentence(sentence))], sequence_max_len, padding='post', value=UNKNOWN_WORD))[0][0]

test_sentence_positive = "This is a great product!"
test_sentence_negative = "That movie was awful."
print("Inferred sentiment (percent positive) for \"" + test_sentence_positive + "\": " + repr(get_sentiment(test_sentence_positive) * 100) + "%\n")
print("Inferred sentiment (percent positive) for \"" + test_sentence_negative + "\": " + repr(get_sentiment(test_sentence_negative) * 100) + "%\n")

#Inspired by code (C) 2017 François Chollet under MIT license, from https://www.tensorflow.org/tutorials/keras/basic_text_classification
