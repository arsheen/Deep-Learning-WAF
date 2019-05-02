from Preprocessing import parse_file, to_string, encode_data, create_vocab_set, load_data
from model import create_model, fit_model, test_model
from __future__ import print_function
from __future__ import division
import json
import numpy as np
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
np.random.seed(123)  # for reproducibility
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # parse the raw data files first
    normal_file_raw = 'dataset/normalTrafficTraining.txt'
    anomaly_file_raw = 'dataset/anomalousTrafficTest.txt'
    normal_test_raw = 'dataset/normalTrafficTest.txt'

    normal_test_parse = 'dataset/normalRequestTest.txt'
    normal_file_parse = 'dataset/normalRequestTraining.txt'
    anomaly_file_parse = 'dataset/anomalousRequestTest.txt'

    # Parse the files to decode the URLs in the raw HTTP requests and write them in a proper format
    parse_file(normal_file_raw, normal_file_parse)
    parse_file(anomaly_file_raw, anomaly_file_parse)
    parse_file(normal_test_raw, normal_test_parse)

    # Convert each HTTP request into a string and append each of these strings to a list
    X_train = to_string('../input/normalRequestTraining.txt')
    X_test_bad = to_string('../input/anomalousRequestTest.txt')
    X_test_good = to_string('../input/normalRequestTest.txt')


    # Label the good requests and bad requests
    # 0 --> good --> [1. 0.]
    # 1 --> bad -->  [0. 1.]
    y_train = [0]*len(X_train)
    y_bad = [1] * len(X_test_bad)
    y_good = [0] * len(X_test_good)

    # Put all the requests in the X and y lists
    y_unshuffled = y_bad+y_good+y_train
    X_unshuffled = X_test_bad + X_test_good + X_train

    # Shuffle the data
    X_shuffled, y_shuffled = shuffle(X_unshuffled, y_unshuffled)
    # use categorical output
    y_shuffled = to_categorical(y_shuffled)


    # set parameters:
    subset = None

    # Maximum length. Longer gets chopped. Shorter gets padded.
    maxlen = 1000

    # Model params
    # Filters for conv layers
    nb_filter = 64
    # Number of units in the dense layer
    dense_outputs = 64
    # Conv layer kernel size
    filter_kernels = [7, 7]
    # Number of units in the final output layer. Number of classes.
    cat_output = 2

    # Compile/fit params
    batch_size = 128
    nb_epoch = 20

    print('Loading data...')
    # # Expect x to be a list of sentences. Y to be index of the categories.
    (xt, yt), (x_test, y_test) = load_data(X_shuffled, y_shuffled)

    print('Creating vocab...')
    vocab, reverse_vocab, vocab_size, alphabet = create_vocab_set()

    print('Compile model...')
    model = create_model(filter_kernels, dense_outputs, maxlen, vocab_size,
                         nb_filter, cat_output)
    # Encode data
    xt = encode_data(xt, maxlen, vocab)
    x_test = encode_data(x_test, maxlen, vocab)

    print('Chars vocab: {}'.format(alphabet))
    print('Chars vocab size: {}'.format(vocab_size))
    print('X_train.shape: {}'.format(xt.shape))
    model.summary()

    print('Fit model...')
    patience = 5 # this is the number of epochs with no improvment after which the training will stop
    history = fit_model(model, xt, yt, patience, batch_size, nb_epoch)

    print("Testing model...")
    score = test_model(x_test, y_test, batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Graphs and data visualisation
    # Training Accuracy Vs validation Accuracy
    plt.figure(0)
    plt.figsize = (10, 10)
    plt.plot(history.history['acc'], 'r')
    plt.plot(history.history['val_acc'], 'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Vs validation Accuracy")
    plt.legend(['train', 'validation'])


    # Training Loss Vs Validation Loss
    plt.figure(0)
    plt.figsize = (10, 10)
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.yticks(np.arange(0, 0.5, 0.1))
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Vs validation Loss")
    plt.legend(['train', 'validation'])

    # Classification Matrix
    y_pred = model.predict(x_test)
    y_pred1 = (y_pred > 0.5)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred1.argmax(axis=1))
    print(matrix)
    plt.matshow(matrix, cmap=plt.cm.gray)
    plt.show()

    row_sum = matrix.sum(axis=1, keepdims=True)
    norm_conf = matrix / row_sum
    print(norm_conf)
    plt.matshow(norm_conf, cmap=plt.cm.gray)
    plt.show()

