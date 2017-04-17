'''
This function is an example to perform sentiment analysis on the imdb
movie dataset using Keras library using TensorFlow as backend.
'''

import time
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

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
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

    print (model.summary())
    return model

def conv1D_model():
    # create conv1D model
    model = Sequential()
    model.add(Embedding(top_words, 32, input_length=max_words))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.2))
    # model.add(Conv1D(filters=48, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(filters=48, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])
    print (model.summary())
    return model


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    # Load only top n words from the dataset
    top_words = 5000
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=top_words)

    max_words = 500
    x_train = sequence.pad_sequences(x_train, maxlen=max_words)
    x_test = sequence.pad_sequences(x_test, maxlen=max_words)

    # model = baseline_model()
    # # fit the model
    # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=128, verbose=2)

    '''
    TO DO:
        Use cross-validation and compute the accuracy
    '''

    start = time.time()
    model = KerasClassifier(
        build_fn=conv1D_model, 
        validation_data=(x_test, y_test),
        epochs=5, 
        batch_size=128, 
        verbose=2)
    model.fit(x_train, y_train)
    # results = cross_val_score(model, x_train, y_train, cv=3)
    # print ("Accuracy: mean {:.2f}, std {:.2f}".format(
    #     results.mean()*100, 
    #     results.std()*100))
    end = time.time()
    elapsed = end - start
    print ("Time taken to fit the model:", time.strftime(
        "%H:%M:%S", 
        time.gmtime(elapsed)))

    # ealuate the model
    # ealuate the model
    y_pred = model.predict(x_test)
    print ("Accuracy:", metrics.accuracy_score(y_test, y_pred))
