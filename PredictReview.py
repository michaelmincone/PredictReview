import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential


#split the data line by line
data = ""
with open('trainingSet.txt', 'r') as file:
    data = file.read().split('\n')

num_reviews = len(data)

#fill up old_train with a tuple (review, rating)
x_train = []
y_train = []
for sentence in data:
    arr = sentence.split('\t')
    if len(arr) == 2:
        word = arr[0]
        rating = arr[1][1]
        x_train.append(word)
        y_train.append(rating)

#strip all punctuation and add the review and rating to their own separate arrays
train = []
for sentence in x_train:
    train.append(sentence.translate(str.maketrans('', '', string.punctuation)).lower())

#take off the trailing space at the end of each sentence
sentences = []
for sentence in train:
    sentences.append(sentence[0:-1])

#get an array of arrays of just the words... ex: [['this', 'food', 'was', 'terrible'], ['awesome', 'food']]
justWords = []
for sentence in sentences:
    justWords.append(sentence.split(' '))

#remove any empty strings in the arrays becuase when trying to vectorize you will get an error
for arr in justWords:
    for word in arr:
        if word == '':
            arr.remove('')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
text_sequences = np.array(tokenizer.texts_to_sequences(sentences))

sequence_dict = tokenizer.word_index
dictionary = dict((num, val) for (val, num) in sequence_dict.items())

# We get a map of encoding-to-word in sequence_dict
# Generate encoded reviews
reviews_encoded = []
for i, review in enumerate(justWords):
    reviews_encoded.append([sequence_dict[x] for x in review])


#we will now add padding to our array to make them all the same size
max = 100;
X = pad_sequences(reviews_encoded, maxlen=max, truncating='post')

# make one hot array for each review
Y = np.array([[0, 1] if '0' in label else [1, 0] for label in y_train])

X_train, Y_train = X[0:450], Y[0:450]
X_val, Y_val = X[451:499], Y[451:499]

model = Sequential()
model.add(Embedding(len(dictionary) + 1, max, input_length=max))
model.add(LSTM(40, return_sequences=True, recurrent_dropout=0.5))
model.add(Dropout(0.5))
model.add(LSTM(40, recurrent_dropout=0.5))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='softmax'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.05, decay=0.1), metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=120, epochs=9, validation_data=(X_val,Y_val))

acc = model.evaluate(X_val, Y_val)

print("validation accuracy is " + str(acc[1]))

