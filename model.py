from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model


def create_model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, cat_output):
    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

    # Define what the input shape looks like
    inputs = Input(shape=(maxlen,), dtype='int64')
    # Use embedding layer
    embedded = Embedding(output_dim=128, input_dim=300, input_length=1000)(inputs)

    # Use Embedding layer as following
    # Think of it as a one-hot embedding and a linear layer mashed into a single layer.
    # See discussion here: https://github.com/keras-team/keras/issues/4838
    # Note this will introduce one extra layer of weights (of size vocab_size x vocab_size = 69*69 = 4761)
    # 300 -> vocab size (using 4 times the vocab size to minimize collisions,
    # 128 -> size of embedding vectors, maxlen -> 1000, size of each request)
    embedded = Embedding(output_dim=128, input_dim=300, input_length=1000)(inputs)

    # All the convolutional layers...
    conv = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[0],
                         padding='valid', activation='relu',
                         input_shape=(maxlen,))(embedded)
    conv = MaxPooling1D(pool_size=7)(conv)

    conv1 = Convolution1D(filters=nb_filter, kernel_size=filter_kernels[1],
                          padding='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_size=40)(conv1)

    conv2 = Flatten()(conv1)

    # Two dense layers with dropout of .5
    x = Dense(dense_outputs, activation='relu')(conv2)
    z = Dropout(0.5)(x)

    # Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)

    model = Model(inputs=inputs, outputs=pred)

    #     sgd = SGD(lr=0.01, momentum=0.9)
    adam = Adam(lr=0.001)  # Feel free to use SGD above. I found Adam with lr=0.001 is faster than SGD with lr=0.01
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def fit_model(model, xt, yt, pat, batch_size, nb_epoch):
    early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)
    model_checkpoint = ModelCheckpoint('clcnn.h5', verbose=1, save_best_only=True)
    history = model.fit(xt, yt, validation_split=0.1, batch_size=batch_size, epochs=nb_epoch, shuffle=True,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=1)
    return history

def test_model(x_test, y_test, batch_size):
    model = load_model('clcnn.h5')
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    return score
