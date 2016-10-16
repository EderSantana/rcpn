#!/usr/bin/env python

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import argparse
import time
from keras import callbacks as cbks
import logging
import numpy as np
from random import sample

from keras import backend as K
from keras.layers import Reshape, TimeDistributed, Input, Convolution2D, Lambda
from keras.models import Model
from layers import ConvRNN
from recurrent_convolutional import RecurrentConv2D as ConvRNN

from utils import prepare_coil100

ims = 32
def coil_gen(data, batch_size=32, m=ims, time_len=10):
    X = np.zeros((batch_size, time_len, 3*m**2)).astype("float32")
    while True:
        X = np.asarray(sample(data, batch_size))
        X = X/127.5 - 1
        # X = X.reshape(batch_size, -1, 3*ims**2)
        yield X[:, :-1], X[:, 1:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ConvRNN model trainer')
    parser.add_argument('--name', type=str, default="conv_rnn", help='Name of the model.')
    # parser.add_argument('--time', type=int, default=1, help='How many temporal frames in a single input.')
    parser.add_argument('--batch', type=int, default=32, help='Batch size.')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--time', type=int, default=10, help='Length of the temporal series')
    parser.add_argument('--game', type=str, default="slime", help='which game to load')
    parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
    parser.add_argument('--loadweights', dest='loadweights', action='store_true', help='Start from checkpoint.')
    parser.set_defaults(skipvalidate=False)
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()

    if not os.path.exists("./outputs/results_"+args.name):
        os.makedirs("./outputs/results_"+args.name)
    if not os.path.exists("./outputs/samples_"+args.name):
        os.makedirs("./outputs/samples_"+args.name)

    nb_filters = 9
    # XX = Input(batch_shape=(args.batch, args.time, 3*ims**2))
    XX = Input(batch_shape=(args.batch, args.time, ims, ims, 3))
    rnn = ConvRNN(nb_filters, 3, 3, reshape_dim=(args.batch, ims, ims, 3), return_sequences=True, batch_input_shape=(args.batch, args.time, 3*ims*ims), consume_less='gpu', dim_ordering='tf', unroll=True)(XX)
    # rnn = ConvRNN(nb_filters, 3, 3, return_sequences=True, batch_input_shape=(args.batch, args.time, ims, ims, 3))(XX)
    print rnn
    out = Lambda(lambda x: K.reshape(x, (args.batch, args.time, ims, ims, nb_filters)), batch_input_shape=(args.batch, args.time, nb_filters*ims*ims), output_shape=(args.time, ims, ims, nb_filters))(rnn)
    print out
    out = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same', dim_ordering='tf', batch_input_shape=(args.batch, ims, ims, nb_filters)))(out)
    print out
    out = Reshape((args.time, 3*ims**2))(out)
    print out
    model = Model(XX, out)
    model.compile("adam", "mse")

    dataset = prepare_coil100(32)[:, :11]
    train_gen = coil_gen(dataset, batch_size=args.batch, time_len=args.time+1)
    valid_gen = coil_gen(dataset, batch_size=args.batch, time_len=args.time+1)
    next(train_gen)
    vX, vY = next(valid_gen)
    print vX

    plt.subplot(121)
    IM0 = plt.imshow(np.random.randn(ims, ims, 3), interpolation='none')
    plt.subplot(122)
    IM1 = plt.imshow(np.random.randn(ims, ims, 3), interpolation='none')
    plt.draw()
    plt.pause(.001)

    for e in range(args.epoch):
        model.fit_generator(train_gen, samples_per_epoch=args.epochsize, nb_epoch=1,
                            validation_data=valid_gen, nb_val_samples=args.batch*5, verbose=1)
        tY = model.predict(vX)
        for i in range(args.time):
            y = vY[0, i]
            t = tY[0, i]
            IM0.set_data(y.reshape(ims, ims, 3))
            IM1.set_data(t.reshape(ims, ims, 3))
            plt.draw()
            plt.pause(.01)
