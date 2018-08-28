import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import dataset


def create_model(vocabulary_size, review_length, word_vector_length=32, dropout=0.2):
    model = Sequential()

    model.add(Embedding(vocabulary_size, word_vector_length, input_length=review_length))
    # The sequence can be treated as one-dimensional spatial data, and CNN might find some relation between the words
    # In general it decreases the training time on LSTM while increasing performance
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # Dropout is needed because LSTMs tend to overfit really fast
    model.add(LSTM(100, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))

    # binary crossentropy or binary log loss because of 2 classes, Adam is more efficient than gradient descent
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


np.random.seed(7)

words_max = 6000

(X_train, y_train), (X_test, y_test) = dataset.load_dataset(max_words=words_max)

max_review_len = 600

X_train = sequence.pad_sequences(X_train, maxlen=max_review_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_len)

word_embedding_vector_length = 32

model = create_model(vocabulary_size=words_max, review_length=max_review_len)

# only 3 epochs because LSTMs overfit really fast
model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)

scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy %.2f%%' % (scores[1] * 100))
