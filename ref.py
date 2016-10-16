#!/usr/bin/env python

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import time
from keras import callbacks as cbks
import logging
import numpy as np
from random import sample

from utils import prepare_coil100
ims = 32


def coil_gen(data, batch_size=32, m=ims, time_len=10):
    X = np.zeros((batch_size, time_len, 3*m**2)).astype("float32")
    while True:
        X = np.asarray(sample(data, batch_size))
        X = X/127.5 - 1
        # X = X.reshape(batch_size, -1, 3*ims**2)
        yield X[:, :-1], X[:, 1:]


def train_model(name, ftrain, generator, samples_per_epoch, nb_epoch,
                verbose=1, callbacks=[], ftest=None,
                validation_data=None, nb_val_samples=None,
                saver=None):
    """
    Main training loop.
    modified from Keras fit_generator
    """
    gif = True
    if gif:
        plt.subplot(121)
        IM = plt.imshow(np.random.randn(ims, ims, 3), interpolation="none")
        plt.subplot(122)
        IM2 = plt.imshow(np.random.randn(ims, ims, 3), interpolation="none")
        plt.draw()
        plt.pause(.001)

    epoch = 0
    counter = 0
    out_labels = ['loss', 'time']  # self.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    history = cbks.History()
    callbacks = [cbks.BaseLogger()] + callbacks + [history]
    if verbose:
        callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)

    callbacks._set_params({
        'nb_epoch': nb_epoch,
        'nb_sample': samples_per_epoch,
        'verbose': verbose,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    while epoch < nb_epoch:
      callbacks.on_epoch_begin(epoch)
      samples_seen = 0
      batch_index = 0
      while samples_seen < samples_per_epoch:
        x, y = next(generator)
        # build batch logs
        batch_logs = {}
        if type(x) is list:
          batch_size = len(x[0])
        elif type(x) is dict:
          batch_size = len(list(x.values())[0])
        else:
          batch_size = len(x)
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_size
        callbacks.on_batch_begin(batch_index, batch_logs)

        t1 = time.time()
        samples, losses = ftrain(x, y, counter)
        outs = (losses, ) + (time.time() - t1, )
        counter += 1

        if (counter % 100 == 0) and gif:
            for v, u in zip(samples[0], y[0]):
                IM.set_data(v.reshape(ims, ims, 3))
                IM2.set_data(u.reshape(ims, ims, 3))
                plt.draw()
                plt.pause(.01)

        for l, o in zip(out_labels, outs):
            batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)

        # construct epoch logs
        epoch_logs = {}
        batch_index += 1
        samples_seen += batch_size

      if validation_data is not None:
          valid_cost = 0
          valid_samples_seen = 0
          while valid_samples_seen < nb_val_samples:
              x, y = next(validation_data)
              valid_cost += ftest(x, y)[1]
              valid_samples_seen += 1
          valid_cost /= float(nb_val_samples)
          print "\nValidation: ", valid_cost

      if saver is not None:
        saver(epoch)

      callbacks.on_epoch_end(epoch, epoch_logs)
      epoch += 1

    # _stop.set()
    callbacks.on_train_end()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generative model trainer')
  parser.add_argument('model', type=str, default="bn_model", help='Model definitnion file')
  parser.add_argument('--name', type=str, default="autoencoder", help='Name of the model.')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
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

  MODEL_NAME = args.model
  logging.info("Importing get_model from {}".format(args.model))
  exec("from models."+MODEL_NAME+" import get_model")
  # try to import `cleanup` from model file

  model_code = open('models/'+MODEL_NAME+'.py').read()

  if not os.path.exists("./outputs/results_"+args.name):
      os.makedirs("./outputs/results_"+args.name)
  if not os.path.exists("./outputs/samples_"+args.name):
      os.makedirs("./outputs/samples_"+args.name)
  if not os.path.exists("./outputs/logs_"+args.name):
      os.system("rm -rf ./outputs/logs_"+args.name)

  with tf.Session() as sess:
    ftrain, ftest, loader, saver, extras = get_model(sess=sess, name=args.name, batch_size=args.batch, time_len=args.time)

    # start from checkpoint
    if args.loadweights:
      loader()

    dataset = prepare_coil100(32)
    print 'Dataset shape:', dataset.shape
    train_gen = coil_gen(dataset[:, :11])
    valid_gen = coil_gen(dataset[:, 11:22])
    train_model(args.name, ftrain,
                train_gen, ftest=ftest, validation_data=valid_gen, nb_val_samples=100,
                samples_per_epoch=args.epochsize,
                nb_epoch=args.epoch, verbose=1, saver=saver
                )
