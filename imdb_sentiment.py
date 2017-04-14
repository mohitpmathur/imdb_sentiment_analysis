'''
This function is an example to perform sentiment analysis on the imdb
movie dataset using Keras library using TensorFlow as backend.
'''

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence

def baseline_model():
    # create model
    model = Sequential()
    model.add(Embedding(top_words, 48, input_length=max_words))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(250, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print (model.summary())
    return model


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    # Load only top n words from the dataset
    top_words = 5000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)

    max_words = 500
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=max_words)

    model = baseline_model()
    # fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128, verbose=2)

    '''
    TO DO:
        Use cross-validation and compute the accuracy
    '''

    # ealuate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print ("Accuracy: {:.2f}".format(scores[1]*100))
