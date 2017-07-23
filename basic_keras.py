#!/usr/bin/env python

import argparse
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from singen import SinGen


lstm_timesteps = 100  # lstm timesteps is how big to train on
lstm_batchsize = 128
lstm_units = 64


class TSModel(object):
    def __init__(self, timesteps, batchsize, stateful=False):
        self.m = Sequential()
        # You can add two of these later
        if stateful:
            bis = (batchsize, timesteps, 1)
            self.m.add(LSTM(lstm_units, return_sequences=True, stateful=stateful,
                            batch_input_shape=bis, input_shape=(timesteps, 1)))
        else:
            self.m.add(LSTM(lstm_units, return_sequences=True, stateful=False,
                            input_shape=(timesteps, 1)))
            self.m.add(LSTM(1, return_sequences=True, stateful=False))
        self.m.compile(loss='mean_squared_error', optimizer=Adam())


def train(m, epochs, lr, batchsize, tensorboard=None, verbose=1):
    m.m.optimizer.lr = lr
    g = SinGen(timesteps=lstm_timesteps, batchsize=batchsize)

    callbacks = None
    if tensorboard is not None:
        callbacks = [TensorBoard(log_dir=tensorboard, histogram_freq=1,
                                 write_graph=True, write_images=True)]
    histories = []
    for i in range(epochs):
        if verbose is not None and verbose >= 1:
            print('------------------------------------------')
            print(i)
            print('------------------------------------------')
        x, y = g.batch()
        h = m.m.fit(x, y, batch_size=lstm_batchsize, epochs=10, verbose=verbose,
                    callbacks=callbacks)
        histories.append(h)

    return histories


def get_args():
    p = argparse.ArgumentParser("Train Keras LSTM Model for sine wave")
    p.add_argument('--save', help="h5 file to save model to when done")
    p.add_argument('--tensorboard', help="tensorboard log dir")
    return p.parse_args()


def main():
    args = get_args()
    m = TSModel(timesteps=lstm_timesteps, batchsize=lstm_batchsize)
    train(m, 128, 5e-3, lstm_batchsize, args.tensorboard)

    if args.save is not None:
        m.m.save_weights(args.save)


if __name__ == '__main__':
    main()
